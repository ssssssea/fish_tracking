# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.47'


from boxmot.tracker_zoo import create_tracker, get_tracker_config

from boxmot.trackers.bytetrack.byte_tracker import BYTETracker





TRACKERS = ['bytetrack']

__all__ = ("__version__",
           "BYTETracker",
           "create_tracker", "get_tracker_config", "gsi")
