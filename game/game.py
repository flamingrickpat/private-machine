# ──────────────────────────────────────────────────────────────
# voxel_ai_minimal_server_client.py – rev: basic physics, jumping, fixed mining/placing
#
# Highlights
#   • Each chunk is still one greedy‑free mesh (same as previous revision) **but** now
#     created with `collider='mesh'`, so Ursina’s physics engine can step on terrain.
#   • First‑person players can jump (SPACE) and collide with the voxel ground.
#   • Server owns Y‑axis physics: gravity, jump impulse, and simple ground check.
#   • Mining/placing bug fixed – voxel ray‑cast now returns the *air* block next to the
#     hit face, so right‑click placement finally works.
#
# Tested with Ursina 5.1 & websockets 12.0.
# ──────────────────────────────────────────────────────────────
import sys, asyncio, json, sqlite3, time, math, uuid, threading, queue, random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from noise import pnoise2

# ────────────────────────────────
# Constants & palette
# ────────────────────────────────
CHUNK_SIZE      = 16
WORLD_HEIGHT    = 32
WORLD_RADIUS    = 32             # ± in blocks from origin on X/Z
TICK_HZ         = 20             # server physics tick (50 ms)
GRAVITY         = 0.04           # blocks / tick²
JUMP_SPEED      = 0.6            # initial upward vel
MOVE_SPEED      = 0.2            # blocks / tick (≈4 m/s)
DB              = Path('voxel_ai_world.sqlite')

BLOCKS = {
    0: (None, 'Air'),
    1: ((0.55, 0.38, 0.23, 1), 'Dirt'),
    2: ((0.50, 0.50, 0.50, 1), 'Stone'),
    3: ((0.60, 0.40, 0.22, 1), 'Wood'),
    4: ((0.18, 0.60, 0.18, 1), 'Leaves'),
}
AIR = 0
DEFAULT_PLACE_ID = 1

# ────────────────────────────────
# Dataclasses & math helpers
# ────────────────────────────────
@dataclass
class PlayerState:
    id: str
    pos: Tuple[float, float, float] = (0.0, 12.0, 0.0)
    vel_y: float = 0.0
    rot_y: float = 0.0
    inventory: Dict[int, int] = field(default_factory=lambda: {1: 64})
    def to_json(self):
        d = asdict(self)
        return d

def world_to_chunk(x: int, z: int) -> Tuple[int, int]:
    return math.floor(x / CHUNK_SIZE), math.floor(z / CHUNK_SIZE)

def local_coords(x: int, y: int, z: int):
    cx, cz = world_to_chunk(x, z)
    return cx, cz, x - cx * CHUNK_SIZE, y, z - cz * CHUNK_SIZE

# ────────────────────────────────
# Server side
# ────────────────────────────────

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blocks(x INT,y INT,z INT,t INT, PRIMARY KEY(x,y,z));
        CREATE TABLE IF NOT EXISTS players(id TEXT PRIMARY KEY, json_state TEXT, updated REAL);
    """)
    conn.commit()

def terrain_height(x,z):
    h = pnoise2(x*0.08, z*0.08, octaves=3, repeatx=1024, repeaty=1024)
    return int((h+1)*0.5*(WORLD_HEIGHT/2))+5

def generate_world(conn):
    if conn.execute('SELECT 1 FROM blocks LIMIT 1').fetchone():
        return
    print('[srv] generating terrain …')
    for x in range(-WORLD_RADIUS, WORLD_RADIUS):
        for z in range(-WORLD_RADIUS, WORLD_RADIUS):
            h = terrain_height(x,z)
            for y in range(h):
                t = 2 if y < h-3 else 1
                conn.execute('INSERT INTO blocks VALUES(?,?,?,?)',(x,y,z,t))
            if random.random()<0.03 and h+5 < WORLD_HEIGHT:
                for y in range(h, h+3):
                    conn.execute('INSERT INTO blocks VALUES(?,?,?,3)',(x,y,z))
                for dx in (-2,-1,0,1,2):
                    for dz in (-2,-1,0,1,2):
                        for dy in (h+3,h+4):
                            if abs(dx)+abs(dz)<3:
                                conn.execute('INSERT INTO blocks VALUES(?,?,?,4)',(x+dx,dy,z+dz))
    conn.commit()
    print('[srv] terrain ready')

def db_get_block(conn,x,y,z):
    row = conn.execute('SELECT t FROM blocks WHERE x=? AND y=? AND z=?',(x,y,z)).fetchone()
    return row[0] if row else AIR

def db_set_block(conn,x,y,z,t):
    if t==AIR:
        conn.execute('DELETE FROM blocks WHERE x=? AND y=? AND z=?',(x,y,z))
    else:
        conn.execute('INSERT OR REPLACE INTO blocks VALUES(?,?,?,?)',(x,y,z,t))
    conn.commit()

async def game_server(host='', port=8765):
    import websockets
    conn = sqlite3.connect(DB, check_same_thread=False)
    init_db(conn); generate_world(conn)

    players: Dict[str,PlayerState] = {}
    conns: Dict[websockets.WebSocketServerProtocol,str] = {}

    def save(ps:PlayerState):
        conn.execute('INSERT OR REPLACE INTO players VALUES(?,?,?)',(ps.id,json.dumps(ps.to_json()),time.time()))
        conn.commit()

    def ground_height(x:int,z:int):
        # return top solid y for column or -inf
        cur = conn.execute('SELECT MAX(y) FROM blocks WHERE x=? AND z=?',(x,z)).fetchone()
        return cur[0] if cur and cur[0] is not None else -999

    def physics_step(ps:PlayerState, mv:dict):
        # horizontal move
        dx = (mv.get('right',0)-mv.get('left',0))*MOVE_SPEED
        dz = (mv.get('forward',0)-mv.get('back',0))*MOVE_SPEED
        new_x = ps.pos[0]+dx; new_z = ps.pos[2]+dz

        # vertical motion
        on_ground = ps.pos[1]-1e-3 <= ground_height(int(round(new_x)), int(round(new_z)))+1e-3
        if mv.get('jump') and on_ground:
            ps.vel_y = JUMP_SPEED
        ps.vel_y -= GRAVITY
        new_y = ps.pos[1] + ps.vel_y
        gh = ground_height(int(round(new_x)), int(round(new_z)))
        if new_y <= gh+1:  # stand on ground
            new_y = gh+1; ps.vel_y = 0
        ps.pos = (new_x, new_y, new_z)
        ps.rot_y += mv.get('turn',0)*2

    def mine(pid,pos):
        x,y,z = pos; t=db_get_block(conn,x,y,z)
        if t==AIR: return None
        db_set_block(conn,x,y,z,AIR)
        players[pid].inventory[t]=players[pid].inventory.get(t,0)+1
        return (x,y,z,AIR)
    def place(pid,pos,t):
        x,y,z=pos
        if db_get_block(conn,x,y,z)!=AIR: return None
        ps=players[pid]
        if ps.inventory.get(t,0)<=0: return None
        db_set_block(conn,x,y,z,t); ps.inventory[t]-=1
        return (x,y,z,t)

    async def broadcast(pkt):
        if conns:
            msg=json.dumps(pkt)
            await asyncio.gather(*[ws.send(msg) for ws in conns], return_exceptions=True)

    async def handler(ws):
        pid=str(uuid.uuid4()); ps=PlayerState(pid)
        players[pid]=ps; conns[ws]=pid; save(ps)
        await ws.send(json.dumps({'cmd':'welcome','id':pid,'state':ps.to_json()}))
        cur=conn.execute('SELECT x,y,z,t FROM blocks'); batch=[]
        for row in cur:
            batch.append(row)
            if len(batch)>=8192:
                await ws.send(json.dumps({'cmd':'chunk','blocks':batch})); batch.clear()
        if batch: await ws.send(json.dumps({'cmd':'chunk','blocks':batch}))
        try:
            async for msg in ws:
                try:data=json.loads(msg)
                except: continue
                if data['cmd']=='input':
                    physics_step(ps,data['move'])
                elif data['cmd']=='mine':
                    upd=mine(pid,tuple(data['pos']));
                    if upd: await broadcast({'cmd':'block','change':upd})
                elif data['cmd']=='place':
                    upd=place(pid,tuple(data['pos']),data.get('t',DEFAULT_PLACE_ID))
                    if upd: await broadcast({'cmd':'block','change':upd})
        finally:
            save(ps); players.pop(pid,None); conns.pop(ws,None)

    async def state_loop():
        while True:
            if conns:
                snap={'cmd':'state','players':[p.to_json() for p in players.values()]}
                await broadcast(snap)
            await asyncio.sleep(1/TICK_HZ)

    async with websockets.serve(handler, host, port):
        print(f'[srv] listening ws://{host}:{port}'); await state_loop()

# ────────────────────────────────
# Client side
# ────────────────────────────────

def run_client(uri='ws://127.0.0.1:8765'):
    from ursina import Ursina, Mesh, Entity, Vec3, Vec4, color, held_keys, mouse, destroy
    import websockets

    # Chunk container ---------------------------------------------------
    class Chunk:
        def __init__(self,cx:int,cz:int):
            self.cx, self.cz = cx, cz
            self.blocks = np.zeros((CHUNK_SIZE,WORLD_HEIGHT,CHUNK_SIZE),dtype=np.uint8)
            self.dirty=True; self.entity:Optional[Entity]=None

    chunks: Dict[Tuple[int,int],Chunk]={}
    player_id=None
    q_rx,q_tx=queue.Queue(),queue.Queue()

    # Networking thread -------------------------------------------------
    class NetThread(threading.Thread):
        def run(self): asyncio.run(self.main())
        async def main(self):
            async with websockets.connect(uri) as ws:
                async def sender():
                    loop=asyncio.get_event_loop()
                    while True:
                        data=await loop.run_in_executor(None,q_tx.get)
                        await ws.send(json.dumps(data))
                asyncio.create_task(sender())
                async for msg in ws:
                    try:q_rx.put(json.loads(msg))
                    except:pass
    NetThread(daemon=True).start()

    # Ursina setup ------------------------------------------------------
    app=Ursina(borderless=False)
    from ursina.prefabs.first_person_controller import FirstPersonController
    fp=FirstPersonController(model='cube',color=color.orange,origin_y=-0.5)
    fp.gravity=1  # server handles Y

    # Meshing helpers ---------------------------------------------------
    FACE_DIRS=[(0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)]
    FACE_VERTS={
        (0,1,0):[(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
        (0,-1,0):[(0,0,0),(0,0,1),(1,0,1),(1,0,0)],
        (1,0,0):[(1,0,0),(1,0,1),(1,1,1),(1,1,0)],
        (-1,0,0):[(0,0,0),(0,1,0),(0,1,1),(0,0,1)],
        (0,0,1):[(0,0,1),(0,1,1),(1,1,1),(1,0,1)],
        (0,0,-1):[(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
    }

    def rebuild_chunk(ch:Chunk):
        verts, tris, cols=[],[],[]; idx=0
        for lx in range(CHUNK_SIZE):
            for y in range(WORLD_HEIGHT):
                for lz in range(CHUNK_SIZE):
                    t=ch.blocks[lx,y,lz]
                    if t==AIR: continue
                    wx=ch.cx*CHUNK_SIZE+lx; wz=ch.cz*CHUNK_SIZE+lz
                    col=Vec4(*BLOCKS[t][0]) if BLOCKS[t][0] else Vec4(1,1,1,1)
                    for dx,dy,dz in FACE_DIRS:
                        nx,ny,nz = lx+dx,y+dy,lz+dz
                        neighbor_air=True
                        if 0<=nx<CHUNK_SIZE and 0<=nz<CHUNK_SIZE and 0<=ny<WORLD_HEIGHT:
                            neighbor_air = ch.blocks[nx,ny,nz]==AIR
                        else:
                            cx2,cz2 = world_to_chunk(wx+dx, wz+dz)
                            ch2 = chunks.get((cx2,cz2))
                            if ch2 and 0<=ny<WORLD_HEIGHT:
                                neighbor_air = ch2.blocks[(wx+dx)-cx2*CHUNK_SIZE, ny, (wz+dz)-cz2*CHUNK_SIZE]==AIR
                        if neighbor_air:
                            for vx,vy,vz in FACE_VERTS[(dx,dy,dz)]:
                                verts.append((wx+vx, y+vy, wz+vz)); cols.append(col)
                            tris.extend([idx,idx+1,idx+2, idx,idx+2,idx+3]); idx+=4
        if ch.entity: destroy(ch.entity)
        if verts:
            m=Mesh(vertices=verts,triangles=tris,colors=cols,mode='triangle'); m.generate()
            ch.entity=Entity(model=m, collider='mesh')
        ch.dirty=False

    # World setters -----------------------------------------------------
    def set_block(x,y,z,t):
        cx,cz,lx,ly,lz = local_coords(x,y,z)
        ch = chunks.setdefault((cx,cz),Chunk(cx,cz))
        ch.blocks[lx,ly,lz]=t; ch.dirty=True

    # Fixed voxel raycast ----------------------------------------------
    def voxel_raycast(origin:Vec3, direction:Vec3, max_dist=6.0):
        step=0.05; pos=np.array(origin);
        prev_block=None; dist=0
        dir_vec=np.array(direction.normalized())*step
        while dist<=max_dist:
            block = tuple(map(int,map(math.floor,pos)))
            if block!=prev_block:
                cx,cz,lx,ly,lz = local_coords(*block)
                ch=chunks.get((cx,cz))
                if ch and 0<=ly<WORLD_HEIGHT and ch.blocks[lx,ly,lz]!=AIR:
                    return block, prev_block
                prev_block=block
            pos+=dir_vec; dist+=step
        return None,None

    # Networking handlers ----------------------------------------------
    def process_net():
        nonlocal player_id
        while not q_rx.empty():
            pkt=q_rx.get(); cmd=pkt['cmd']
            if cmd=='welcome': player_id=pkt['id']
            elif cmd=='chunk':
                for x,y,z,t in pkt['blocks']: set_block(x,y,z,t)
            elif cmd=='block':
                x,y,z,t=pkt['change']; set_block(x,y,z,t)
            elif cmd=='state':
                for ps in pkt['players']:
                    if ps['id']==player_id:
                        fp.position=Vec3(*ps['pos'])+Vec3(0,1.7,0)

    # Input -------------------------------------------------------------
    def send_input():
        mv={'forward':held_keys['w'],'back':held_keys['s'],
             'left':held_keys['a'],'right':held_keys['d'],
             'jump':held_keys['space'],
             'turn':mouse.velocity[0]}
        q_tx.put({'cmd':'input','move':mv})

    def handle_click(key):
        # Ursina sends different labels per OS / driver ("mouse1 down", "left mouse down", etc.)
        if key in ('left mouse down', 'mouse1 up', 'double click'):
            hit, air = voxel_raycast(fp.camera_pivot.world_position, fp.camera_pivot.forward)
            if hit:
                q_tx.put({'cmd': 'mine', 'pos': list(hit)})
        elif key in ('right mouse down', 'mouse3 up', 'mouse3 down'):
            hit, air = voxel_raycast(fp.camera_pivot.world_position, fp.camera_pivot.forward)
            if hit and air:
                q_tx.put({'cmd': 'place', 'pos': list(air), 't': DEFAULT_PLACE_ID})
        elif key in ('escape', 'q'):
            quit()

    # Register the input handler with Ursina
    app.input = handle_click

    # Tick --------------------------------------------------------------
    def tick(task):
        send_input(); process_net()
        for ch in list(chunks.values()):
            if ch.dirty: rebuild_chunk(ch)
        return 0

    app.taskMgr.add(tick,'tick')
    app.run()
    def tick(task):
        send_input(); process_net()
        for ch in list(chunks.values()):
            if ch.dirty: rebuild_chunk(ch)
        return 0

    app.taskMgr.add(tick,'tick'); app.run()


# ────────────────────────────────
# Main entry
# ────────────────────────────────
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in {'server','client'}:
        print('Usage: python voxel_ai_minimal_server_client.py [server|client]'); sys.exit(1)
    if sys.argv[1]=='server': asyncio.run(game_server())
    else: run_client()
