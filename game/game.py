# ──────────────────────────────────────────────────────────────
# voxel_ai_minimal_server_client.py  –  revision: basic terrain & blocks
#
# Adds **early‑Minecraft mechanics**:
#   • Procedural terrain (noise height‑map) with dirt, stone, trees
#   • Mine block – left‑click; Place from inventory – right‑click
#   • Player inventory persists in SQLite alongside world blocks
#   • Server keeps authoritative world; clients render coloured cubes
#
# Requirements:  pip install ursina websockets numpy noise
# ──────────────────────────────────────────────────────────────

import sys, asyncio, json, sqlite3, time, math, uuid, threading, queue, random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Tuple, List

# External libs imported lazily to allow head‑less server
from noise import pnoise2            # Perlin noise

# ────────────────────────────────
# Block palette (id → colour)
# ────────────────────────────────
BLOCKS = {
    0: (None, None),                                # air
    1: ((0.55,0.38,0.23,1), 'Dirt'),
    2: ((0.5,0.5,0.5,1),    'Stone'),
    3: ((0.60,0.40,0.22,1), 'Wood'),
    4: ((0.18,0.6,0.18,1),  'Leaves'),
}
DEFAULT_PLACE_ID = 1

# ────────────────────────────────
# Dataclasses
# ────────────────────────────────
@dataclass
class PlayerState:
    id: str
    pos: Tuple[float,float,float] = (0.0, 10.0, 0.0)
    rot_y: float = 0.0
    inventory: Dict[int,int] = field(default_factory=lambda:{1:10})
    def to_json(self):
        return asdict(self)

# ────────────────────────────────
# SECTION 1 – SERVER
# ────────────────────────────────
WORLD_RADIUS = 32          # blocks (x,z from –32..31)
WORLD_HEIGHT = 32          # y 0..31
DB = Path('voxel_ai_world.sqlite')

def init_db(conn:sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS blocks(x INT, y INT, z INT, t INT,
                                      PRIMARY KEY(x,y,z));
    CREATE TABLE IF NOT EXISTS players(id TEXT PRIMARY KEY,
                                       json_state TEXT, updated REAL);
    """)
    conn.commit()

# ------------- terrain generation -------------

def terrain_height(x:int,z:int)->int:
    # Perlin noise heightmap scaled to WORLD_HEIGHT/2 .. WORLD_HEIGHT-4
    h = pnoise2(x*0.08, z*0.08, octaves=3, repeatx=1024, repeaty=1024)
    return int((h+1)*0.5*(WORLD_HEIGHT/2))+5

def generate_world(conn:sqlite3.Connection):
    cur=conn.execute('SELECT COUNT(*) FROM blocks');
    if cur.fetchone()[0]:
        return  # already generated
    print('[srv] generating terrain ...')
    for x in range(-WORLD_RADIUS, WORLD_RADIUS):
        for z in range(-WORLD_RADIUS, WORLD_RADIUS):
            h=terrain_height(x,z)
            for y in range(h):
                t = 2 if y < h-3 else 1     # stone below, dirt top 3
                conn.execute('INSERT INTO blocks VALUES(?,?,?,?)',(x,y,z,t))
            # tree?
            if random.random()<0.03 and h+5<WORLD_HEIGHT:
                for y in range(h, h+3):
                    conn.execute('INSERT INTO blocks VALUES(?,?,?,3)',(x,y,z))
                for dx in (-2,-1,0,1,2):
                    for dz in (-2,-1,0,1,2):
                        for dy in (h+3,h+4):
                            if abs(dx)+abs(dz)<3:
                                conn.execute('INSERT INTO blocks VALUES(?,?,?,4)',(x+dx,dy,z+dz))
    conn.commit()
    print('[srv] terrain ready')

# ------------- helpers -------------

def db_get_block(conn,x,y,z):
    cur=conn.execute('SELECT t FROM blocks WHERE x=? AND y=? AND z=?',(x,y,z))
    row=cur.fetchone();
    return row[0] if row else 0

def db_set_block(conn,x,y,z,t):
    if t==0:
        conn.execute('DELETE FROM blocks WHERE x=? AND y=? AND z=?',(x,y,z))
    else:
        conn.execute('INSERT OR REPLACE INTO blocks VALUES(?,?,?,?)',(x,y,z,t))
    conn.commit()

async def game_server(host='', port=8765, tick_hz=20):
    import websockets
    conn = sqlite3.connect(DB, check_same_thread=False)
    init_db(conn)
    generate_world(conn)

    players: Dict[str,PlayerState] = {}
    connections: Dict[websockets.WebSocketServerProtocol,str] = {}

    # ---- player helpers ----
    def save_player(ps:PlayerState):
        conn.execute('INSERT OR REPLACE INTO players VALUES(?,?,?)',
                     (ps.id, json.dumps(ps.to_json()), time.time()))
        conn.commit()

    def handle_mine(pid:str, pos:Tuple[int,int,int]):
        x,y,z = pos
        t = db_get_block(conn,x,y,z)
        if t==0:
            return None
        db_set_block(conn,x,y,z,0)
        ps=players[pid]
        ps.inventory[t]=ps.inventory.get(t,0)+1
        return (x,y,z,0)

    def handle_place(pid:str, pos:Tuple[int,int,int], t:int):
        x,y,z=pos
        if db_get_block(conn,x,y,z)!=0:
            return None
        ps=players[pid]
        if ps.inventory.get(t,0)<=0:
            return None
        db_set_block(conn,x,y,z,t)
        ps.inventory[t]-=1
        return (x,y,z,t)

    # ---- websocket handlers ----
    async def client_handler(ws):
        pid=str(uuid.uuid4()); ps=PlayerState(pid)
        players[pid]=ps; connections[ws]=pid; save_player(ps)
        # send welcome + initial block dump in small chunks
        await ws.send(json.dumps({'cmd':'welcome','id':pid,'state':ps.to_json()}))
        blocks=[]
        cur=conn.execute('SELECT x,y,z,t FROM blocks')
        for row in cur:
            blocks.append(row)
            if len(blocks)>=4096:
                await ws.send(json.dumps({'cmd':'chunk','blocks':blocks}))
                blocks=[]
        if blocks:
            await ws.send(json.dumps({'cmd':'chunk','blocks':blocks}))
        try:
            async for msg in ws:
                try:data=json.loads(msg)
                except json.JSONDecodeError:continue
                if data.get('cmd')=='input':
                    move=data.get('move',{})
                    dx=(move.get('right',0)-move.get('left',0))*0.2
                    dz=(move.get('forward',0)-move.get('back',0))*0.2
                    ps.pos=(ps.pos[0]+dx,ps.pos[1],ps.pos[2]+dz)
                    ps.rot_y+=move.get('turn',0)*2
                elif data.get('cmd')=='mine':
                    upd=handle_mine(pid,tuple(data['pos']))
                    if upd:
                        await broadcast({'cmd':'block','change':upd})
                elif data.get('cmd')=='place':
                    upd=handle_place(pid,tuple(data['pos']),data.get('t',DEFAULT_PLACE_ID))
                    if upd:
                        await broadcast({'cmd':'block','change':upd})
        finally:
            print('[srv] disconnect',pid); save_player(ps)
            players.pop(pid,None); connections.pop(ws,None)

    async def broadcast(pkt):
        if not connections: return
        msg=json.dumps(pkt)
        await asyncio.gather(*[ws.send(msg) for ws in connections])

    async def state_loop():
        interval=1/tick_hz
        while True:
            if connections:
                snap={ 'cmd':'state','players':[p.to_json() for p in players.values()] }
                await broadcast(snap)
            await asyncio.sleep(interval)

    async with websockets.serve(client_handler, host, port):
        print(f'[srv] listening ws://{host}:{port}')
        await state_loop()

# ────────────────────────────────
# SECTION 2 – CLIENT
# ────────────────────────────────

def run_client(uri='ws://127.0.0.1:8765'):
    from ursina import Ursina, Entity, Vec3, Vec4, color, mouse, held_keys, raycast, destroy
    from ursina.prefabs.first_person_controller import FirstPersonController
    import websockets

    q_rx, q_tx = queue.Queue(), queue.Queue()

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

    app=Ursina()
    fp=FirstPersonController(model='cube',color=color.orange,origin_y=-0.5)
    fp.gravity=0
    Entity(model='plane',scale=200,texture='white_cube',texture_scale=(200,200),color=color.gray)

    # ---- per-frame network tick ----
    def _net_tick(task):
        send_input(); handle_net();
        return 0  # continue
    app.taskMgr.add(_net_tick, 'net_tick')

    player_id=None
    cubes:{}={}

    def send_input():
        mv={ 'forward':held_keys['w'],'back':held_keys['s'],
             'left':held_keys['a'],'right':held_keys['d'],
             'turn':mouse.velocity[0] }
        q_tx.put({'cmd':'input','move':mv})

    def voxel_color(t:int):
        return Vec4(*BLOCKS[t][0]) if t in BLOCKS else color.white

    def add_cube(x,y,z,t):
        if (x,y,z) in cubes: destroy(cubes[(x,y,z)])
        if t==0:
            cubes.pop((x,y,z),None)
            return
        e=Entity(model='cube',position=(x,y,z),scale=1,color=voxel_color(t),collider='box')
        cubes[(x,y,z)]=e

    def handle_net():
        nonlocal player_id
        while not q_rx.empty():
            pkt=q_rx.get()
            if pkt['cmd']=='welcome': player_id=pkt['id']
            elif pkt['cmd']=='chunk':
                for x,y,z,t in pkt['blocks']: add_cube(x,y,z,t)
            elif pkt['cmd']=='block':
                x,y,z,t=pkt['change']; add_cube(x,y,z,t)
            elif pkt['cmd']=='state':
                for ps in pkt['players']:
                    if ps['id']==player_id:
                        fp.position=Vec3(*ps['pos'])+Vec3(0,1.7,0)

    def input(key):
        if key=='left mouse down':
            hit=raycast(fp.camera_pivot.world_position,fp.camera_pivot.forward,distance=5,ignore=[fp,])
            if hit.entity:
                pos=[int(round(c)) for c in hit.entity.position]
                q_tx.put({'cmd':'mine','pos':pos})
        if key=='right mouse down':
            hit=raycast(fp.camera_pivot.world_position,fp.camera_pivot.forward,distance=5,ignore=[fp,])
            if hit.entity:
                normal=[int(n) for n in hit.normal]
                pos=[int(round(a+b)) for a,b in zip(hit.entity.position,normal)]
                q_tx.put({'cmd':'place','pos':pos,'t':DEFAULT_PLACE_ID})
        if key=='escape': quit()

    def update():
        send_input(); handle_net()

    import builtins; builtins.update = update  # expose update for Ursina if needed
    app.run()

# ────────────────────────────────
# ENTRY
# ────────────────────────────────
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in {'server','client'}:
        print('usage: python voxel_ai_minimal_server_client.py [server|client]');sys.exit(1)
    if sys.argv[1]=='server':
        asyncio.run(game_server())
    else:
        run_client()
