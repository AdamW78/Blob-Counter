import enum
import logging
from collections import namedtuple

Action = namedtuple('Action', ['action_type', 'keypoint'])

class UndoRedoTracker:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def perform_action(self, action: Action):
        """Perform an action and add it to the undo stack."""
        if action is None:
            return # No action to perform
        self.undo_stack.append(action)
        self.redo_stack.clear()  # Clear the redo stack as new action invalidates the redo history
        logging.debug(f"Action performed: Action[ActionType={action.action_type}, Keypoint=(x={action.keypoint.pt[0]}, y={action.keypoint.pt[1]})]")

    def undo(self):
        """Undo the last action."""
        if not self.undo_stack:
            logging.debug("No undo stack")
            return  # No action to undo

        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        logging.debug(f"Undoing action: Action[ActionType={action.action_type}, Keypoint=(x={action.keypoint.pt[0]}, y={action.keypoint.pt[1]})]")
        return action


    def redo(self):
        """Redo the last undone action."""
        if not self.redo_stack:
            logging.debug("No redo stack")
            return  # No action to redo

        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        logging.debug(f"Redoing action: Action[ActionType={action.action_type}, Keypoint=(x={action.keypoint.pt[0]}, y={action.keypoint.pt[1]})]")
        return action

class ActionType(enum.Enum):
    ADD = 0,
    REMOVE = 1
