"""Engine-assisted text observations shared by all backends."""

import math

import numpy as np

from .specs import audio_description


class Perception:
    def __init__(self):
        self.history = []
        self.previous = None
        self.visits = {}
        self.messages = []

    def build(self, task, state, variables, audio, notifications):
        x, y, angle = (variables[key] for key in ("POSITION_X", "POSITION_Y", "ANGLE"))
        position = (round(x / 64), round(y / 64))
        self.visits[position] = self.visits.get(position, 0) + 1
        if notifications:
            for message in notifications:
                if message.startswith("Shoot:") and message not in self.messages:
                    self.messages.append(message)
        h, w = state.screen_buffer.shape[:2]
        objects = []
        for label in state.labels:
            if label.object_category == "Self":
                continue
            # Basic Audio must not reveal invisible target positions through engine labels.
            if task.name == "basic_audio" and label.object_category == "Monster":
                continue
            if not np.any(state.labels_buffer == label.value):
                continue
            dx, dy = label.object_position_x - x, label.object_position_y - y
            bearing = (math.degrees(math.atan2(dy, dx)) - angle + 180) % 360 - 180
            centered = label.x - 6 <= w / 2 <= label.x + label.width + 6
            objects.append(
                {
                    "type": label.object_name,
                    "category": label.object_category,
                    "screen_x": round((label.x + label.width / 2) / w, 3),
                    "screen_y": round((label.y + label.height / 2) / h, 3),
                    "bearing_left_positive_deg": round(bearing, 1),
                    "distance_engine_units": round(math.hypot(dx, dy), 1),
                    "aim": "aligned horizontally"
                    if centered
                    else "left"
                    if bearing > 0
                    else "right",
                    "velocity_xy": [
                        round(label.object_velocity_x, 1),
                        round(label.object_velocity_y, 1),
                    ],
                }
            )
        objects.sort(key=lambda item: item["distance_engine_units"])
        band = state.depth_buffer[h // 2 - 12 : h // 2 + 12]
        depth = {
            "left": int(np.median(band[:, : w // 5])),
            "ahead": int(np.median(band[:, w // 2 - 20 : w // 2 + 20])),
            "right": int(np.median(band[:, -w // 5 :])),
        }
        trend = (
            None
            if self.previous is None
            else {
                "health_change": variables["HEALTH"] - self.previous["HEALTH"],
                "distance_moved": round(
                    math.hypot(x - self.previous["POSITION_X"], y - self.previous["POSITION_Y"]), 1
                ),
            }
        )
        self.previous = variables.copy()
        return {
            "mission": task.goal,
            "player": variables,
            "changes_since_last_action": trend,
            "visible_objects": objects[:32],
            "depth_larger_is_more_room": depth,
            "recent_actions": self.history[-8:],
            "visits_to_current_64_unit_cell": self.visits[position],
            "notifications": self.messages[-8:],
            "audio": audio_description(audio) if task.name == "basic_audio" else None,
            "perception_mode": "engine-assisted text, not raw-pixel inference",
        }
