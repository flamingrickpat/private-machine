"""Verify baseline body identities and a reversible camera action against real Minecraft."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, JsonValue

from workflows.minecraft_body.mcp_client import MinecraftToolReply, MinecraftTransport, MinecraftTransportSettings


class BodySetup(BaseModel):
    participant: str
    username: str
    server: str
    endpoint: str


class WorldSetup(BaseModel):
    scenario: str
    seed: int
    bodies: list[BodySetup]


class MinecraftSetup(BaseModel):
    worlds: list[WorldSetup]

    @classmethod
    def load(cls, path: Path) -> MinecraftSetup:
        return cls.model_validate_json(path.read_bytes())


class BodyIdentity(BaseModel):
    username: str
    server: str
    connected: bool
    spawned: bool
    schema_version: str


class Camera(BaseModel):
    yaw: float


class ObservedPose(BaseModel):
    camera: Camera


def checked_result(reply: MinecraftToolReply) -> JsonValue:
    if not reply["ok"]:
        raise RuntimeError(f"Minecraft {reply['tool']} failed: {reply['result']}")
    return reply["result"]


@dataclass(slots=True)
class BodyPreflight:
    setup: BodySetup
    transport: MinecraftTransport

    def observe_yaw(self) -> float:
        reply = self.transport.call("minecraft_observe", {"include_image": False})  # pm-lint: allow PM005 -- external MCP observation arguments
        return ObservedPose.model_validate(checked_result(reply)).camera.yaw

    def rotate_to(self, yaw: float) -> None:
        reply = self.transport.call("minecraft_rotate", {"absolute": True, "yaw_degrees": yaw})  # pm-lint: allow PM005 -- external MCP rotation arguments
        checked_result(reply)

    def check_identity(self) -> BodyIdentity:
        reply = self.transport.call("minecraft_info", {})  # pm-lint: allow PM005 -- external MCP tool with no parameters
        identity = BodyIdentity.model_validate(checked_result(reply))
        if not identity.connected or not identity.spawned:
            raise RuntimeError(f"{identity.username} is not spawned on {identity.server}.")
        if identity.username != self.setup.username or identity.server != self.setup.server:
            raise ValueError(f"{self.setup.endpoint} identifies {identity.username} on {identity.server}; expected {self.setup.username} on {self.setup.server}.")
        return identity

    def check_motion(self) -> None:
        """Verify a quarter turn by observing it, then return the disposable body to its initial heading."""
        original = self.observe_yaw()
        target = (original + 90) % 360
        try:
            self.rotate_to(target)
            observed = self.observe_yaw()
            if abs((observed - target + 180) % 360 - 180) > 2:
                raise RuntimeError(f"{self.setup.username} facing {observed} degrees after requesting {target}.")
        finally:
            self.rotate_to(original)


def prepare_body(setup: BodySetup) -> BodyPreflight:
    settings = MinecraftTransportSettings(endpoint=setup.endpoint)
    return BodyPreflight(setup, MinecraftTransport(settings, model=None))
