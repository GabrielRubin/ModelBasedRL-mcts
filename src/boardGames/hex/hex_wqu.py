from enum import Enum
from typing import Tuple

class HexDirection(Enum):
    VERTICAL   = 0
    HORIZONTAL = 1

class WQuickUnion:
    def __init__(self, size:int, direction:HexDirection):
        self.parent    = [i for i in range(0, (size * size) + 2)]
        self.tree_size = [1 for i in range(0, (size * size) + 2)]
        self.size      = size

        if direction is HexDirection.HORIZONTAL:
            for y in range(0, size):
                self.union(self.get_index_from_pos(0, y), -1)
                self.union(self.get_index_from_pos(self.size-1, y), -2)
        else:
            for x in range(0, size):
                self.union(self.get_index_from_pos(x, 0), -1)
                self.union(self.get_index_from_pos(x, self.size-1), -2)

    def find(self, index:int):
        while index != self.parent[index]:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def check_win(self):
        return self.connected(-1, -2)

    def connected(self, index_a:int, index_b:int):
        return self.find(index_a) == self.find(index_b)

    def union(self, index_a:int, index_b:int):
        root_a = self.find(index_a)
        root_b = self.find(index_b)
        if root_a == root_b:
            return
        if self.tree_size[root_a] < self.tree_size[root_b]:
            self.parent[root_a] = root_b
            self.tree_size[root_b] += self.tree_size[root_a]
        else:
            self.parent[root_b] = root_a
            self.tree_size[root_a] += self.tree_size[root_b]

    def connect_points(self, point_a:Tuple[int,int], point_b:Tuple[int,int]):
        index_a = self.get_index_from_pos(point_a[0], point_a[1])
        index_b = self.get_index_from_pos(point_b[0], point_b[1])
        self.union(index_a, index_b)

    def get_index_from_pos(self, x:int, y:int):
        #assert x >= 0 and x < self.size and y >= 0 and y < self.size #TEST
        index = y * self.size + x
        #assert self.is_index_valid(index) #TEST
        return index

    def is_index_valid(self, index:int):
        if index < 0 or index >= len(self.parent):
            return False
        return True

    def __repr__(self):
        return 'Win: {0}'.format(self.check_win())
