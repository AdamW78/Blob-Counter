import enum


class UndoRedoTracker(object):
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push(self, action):
        self.undo_stack.append(action)
        self.redo_stack = []

    def undo(self):
        if not self.undo_stack:
            return None
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        return action

    def redo(self):
        if not self.redo_stack:
            return None
        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        return action

    def clear(self):
        self.undo_stack = []
        self.redo_stack = []


class ActionType(enum.Enum):
    ADD = 0,
    REMOVE = 1
