"""
@author: Bhrugvish Vakil
@email: vakilb1@udayton.edu
@date: 05-16-2020
"""
import queue
import time
from game import Node

def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def pos0268(pos, puz, parent_dict, q):
    child_1 = puz.copy()
    child_2 = puz.copy()

    if pos == 0:
        swapPositions(child_1, 0, 1)
        swapPositions(child_2, 0, 3)
    elif pos == 2:
        swapPositions(child_1, 2, 1)
        swapPositions(child_2, 2, 5)
    elif pos == 6:
        swapPositions(child_1, 6, 3)
        swapPositions(child_2, 6, 7)
    elif pos == 8:
        swapPositions(child_1, 8, 5)
        swapPositions(child_2, 8, 7)


    cur_key = ''.join(str(i) for i in puz)
    child_1_key = ''.join(str(i) for i in child_1)
    child_2_key = ''.join(str(i) for i in child_2)

    #add child_1
    if child_1_key not in parent_dict:
        q.put(child_1) #push the new child into queue
        parent_dict.update({child_1_key: cur_key}) #map the child-parent relationship

    #add child_2
    if child_2_key not in parent_dict:
        q.put(child_2) #push the new child into queue
        parent_dict.update({child_2_key: cur_key}) #map the child-parent relationship


def pos1357(pos, puz, parent_dict, q):
    child_1 = puz.copy()
    child_2 = puz.copy()
    child_3 = puz.copy()

    if pos == 1:
        swapPositions(child_1, 1, 0)
        swapPositions(child_2, 1, 2)
        swapPositions(child_3, 1, 4)
    elif pos == 3:
        swapPositions(child_1, 3, 0)
        swapPositions(child_2, 3, 6)
        swapPositions(child_3, 3, 4)
    elif pos == 5:
        swapPositions(child_1, 5, 2)
        swapPositions(child_2, 5, 8)
        swapPositions(child_3, 5, 4)
    elif pos == 7:
        swapPositions(child_1, 7, 6)
        swapPositions(child_2, 7, 8)
        swapPositions(child_3, 7, 4)

    cur_key = ''.join(str(i) for i in puz)
    child_1_key = ''.join(str(i) for i in child_1)
    child_2_key = ''.join(str(i) for i in child_2)
    child_3_key = ''.join(str(i) for i in child_3)

    # add child_1
    if child_1_key not in parent_dict:
        q.put(child_1)  # push the new child into queue
        parent_dict.update({child_1_key: cur_key})  # map the child-parent relationship

    # add child_2
    if child_2_key not in parent_dict:
        q.put(child_2)  # push the new child into queue
        parent_dict.update({child_2_key: cur_key})  # map the child-parent relationship

    # add child_3
    if child_3_key not in parent_dict:
        q.put(child_3)  # push the new child into queue
        parent_dict.update({child_3_key: cur_key})  # map the child-parent relationship


def pos4(pos, puz, parent_dict, q):
    child_1 = puz.copy()
    child_2 = puz.copy()
    child_3 = puz.copy()
    child_4 = puz.copy()

    swapPositions(child_1, 4, 1)
    swapPositions(child_2, 4, 3)
    swapPositions(child_3, 4, 5)
    swapPositions(child_4, 4, 7)

    cur_key = ''.join(str(i) for i in puz)
    child_1_key = ''.join(str(i) for i in child_1)
    child_2_key = ''.join(str(i) for i in child_2)
    child_3_key = ''.join(str(i) for i in child_3)
    child_4_key = ''.join(str(i) for i in child_4)

    # add child_1
    if child_1_key not in parent_dict:
        q.put(child_1)  # push the new child into queue
        parent_dict.update({child_1_key: cur_key})  # map the child-parent relationship

    # add child_2
    if child_2_key not in parent_dict:
        q.put(child_2)  # push the new child into queue
        parent_dict.update({child_2_key: cur_key})  # map the child-parent relationship

    # add child_3
    if child_3_key not in parent_dict:
        q.put(child_3)  # push the new child into queue
        parent_dict.update({child_3_key: cur_key})  # map the child-parent relationship

    # add child_4
    if child_4_key not in parent_dict:
        q.put(child_4)  # push the new child into queue
        parent_dict.update({child_4_key: cur_key})  # map the child-parent relationship


#define a dictionary to map position to children
potential_children = {
    0: pos0268,
    1: pos1357,
    2: pos0268,
    3: pos1357,
    4: pos4,
    5: pos1357,
    6: pos0268,
    7: pos1357,
    8: pos0268

}


def dfs(puzzle):
    start = time.time()
    root = puzzle_graph(puzzle)
    get, path = root.search()
    if get:
        list = path
        end = time.time()
        print("Time=", (end - start) * 1000, "ms")
    else:
        list = []
        end = time.time()
        print("Time=", (end - start) * 1000, "ms")
        print("Exceed the maximum steps, please resize the max")

    return list


def bfs(puzzle):
    visited_list = []
    path = []
    parent_dict = {}
    goal_puz = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    goal_key = ''.join(str(i) for i in goal_puz)
    init_key = ''.join(str(i) for i in puzzle)

    q = queue.Queue()
    q.put(puzzle)


    #perform BFS
    while not q.empty():
        cur = q.get()
        if cur == goal_puz:
            break


        pos = 0
        for i in range(9):
            if(cur[i] == 8):
                pos = i

        potential_children[pos](pos, cur,  parent_dict, q)



    #trace back the path
    parent_key = parent_dict[goal_key]
    path.append(goal_key.find('8'))
    path.append(parent_key.find('8'))

    while not parent_key == init_key:
        parent_key = parent_dict[parent_key]
        path.append(parent_key.find('8'))

    path.reverse()
    return path

class puzzle_graph(object):
    max = 15
    def __init__(self, array, path=None):
        if path is None:
            path = []
        self.arr = array
        self.path = path

    def sub_state(self,current):
        sub = []
        if current % 3 != 0:  # 左移
            puz = self.arr.copy()
            puz[current] = puz[current - 1]
            puz[current - 1] = 8
            temp = self.path.copy()
            temp.append(current - 1)
            new = puzzle_graph(puz, temp)
            sub.append(new)

        if current % 3 != 2:  # 右移
            puz = self.arr.copy()
            puz[current] = puz[current + 1]
            puz[current + 1] = 8
            temp = self.path.copy()
            temp.append(current + 1)
            new = puzzle_graph(puz, temp)
            sub.append(new)

        if current // 3 != 0:  # 上移
            puz = self.arr.copy()
            puz[current] = puz[current - 3]
            puz[current - 3] = 8
            temp = self.path.copy()
            temp.append(current - 3)
            new = puzzle_graph(puz, temp)
            sub.append(new)

        if current // 3 != 2:  # 下移
            puz = self.arr.copy()
            puz[current] = puz[current + 3]
            puz[current + 3] = 8
            temp = self.path.copy()
            temp.append(current + 3)
            new = puzzle_graph(puz, temp)
            sub.append(new)

        return sub

    def search(self):
        stack = []
        used = []
        stack.append(self)
        path = []
        if self.arr == [0,1,2,3,4,5,6,7,8]:
            return True, self.path, self
        while True:
            if not stack:
                return False, path
            node = stack.pop()
            used.append(node)
            if self.sort_by(node) == 9:
                return True, node.path
            i = 0
            while i < 9:
                if node.arr[i] == 8:
                    current = i
                    break
                i += 1
            s = node.sub_state(current)
            if s:
                que = sorted(s, key=self.sort_by)
            else:
                continue
            for x in que:
                if len(x.path) < puzzle_graph.max:
                    if self.check(x, used):
                        continue
                    stack.append(x)

    def sort_by(self, state):#排序
        final = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        index = 0
        len = 0
        while index < 9:
            if state.arr[index] == final[index]:
                len += 1
            index+=1
        return len

    def check(self, state1, stop):
        for x in stop:
            if state1.arr == x:
                return True
        return False


def astar(puzzle):
    start = time.time()
    puzzle = puzzle
    goal = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    open_stack = []
    closed_stack = []
    sol_stack = []

    root = Node(None, puzzle, 0)

    open_stack.append(root)
    itr = 0
    list = []
    while len(open_stack) != 0:
        last_index = len(open_stack) - 1
        curr_node = open_stack[last_index]

        closed_stack.append(open_stack[last_index])
        open_stack.remove(open_stack[last_index])
        if curr_node.puzzle == goal:
            # print("goal")
            print("Goal found in ", (time.time() - start), "seconds .")
            path_node = curr_node

            # self.sol_stack.append(curr_node.puzzle)
            while path_node.parent is not None:

                for i in range(0, 9):
                    if path_node.puzzle[i] == 8:
                        list.insert(0, i)

                sol_stack.append(path_node.puzzle)
                path_node = path_node.parent

            # print("list ",(list))
            # print("Sol path: %d" % len(self.sol_stack))
            # print("Sol path: %s" % self.sol_stack)
            return list
        possible_children = possible_moves(curr_node.puzzle)

        for t_child in possible_children:
            if should_be_added(t_child[0]):
                child = Node(curr_node, t_child[0], t_child[1])
                curr_node.children.append(child)

        curr_node.sort_children()
        for child in curr_node.children:
            open_stack.append(child)
        itr += 1


def should_be_added(self, puzzle):
    is_present_in_open_stack = False
    for node in self.open_stack:
        if puzzle == node.puzzle:
            is_present_in_open_stack = True
            break

    is_present_in_closed_stack = False
    for node in self.closed_stack:
        if puzzle == node.puzzle:
            is_present_in_closed_stack = True
            break

    if not is_present_in_open_stack and not is_present_in_closed_stack:
        return True
    return False


def possible_moves(self, puzzle):
    index = puzzle.index(8)
    # print("index of 9 ", index)

    row = int(index / 3)
    col = int(index % 3)
    # print("Postion of 9 ", row, col)
    moves = [1, -1, -1, 1]
    x = 0
    possible_indexs = []
    for i in moves:
        si = -1

        if x % 2 == 0:
            if row + i >= 0 and row + i <= 2:
                si = (row + i) * 3 + col

        else:
            if col + i >= 0 and col + i <= 2:
                si = row * 3 + (col + i)
        x += 1
        if si >= 0 and si <= 8:
            possible_indexs.append(si)

    temp = []

    if puzzle[0] == 0 and puzzle[1] == 1 and puzzle[2] == 2:
        for i in possible_indexs:
            if i >= 0 and i <= 2:
                possible_indexs.remove(i)
    if puzzle[0] == 0 and puzzle[3] == 3 and puzzle[6] == 6:
        for i in possible_indexs:
            if i % 3 == 0:
                possible_indexs.remove(i)

    for i in possible_indexs:
        # print("Index and I",index , int(i))
        p = self.swap_with_blank(deepcopy(puzzle), index, int(i))
        # print("Swaped puzzle ",p)
        temp.append(p)
    # print("List of all possible moves : ",temp)
    return temp


def swap_with_blank(self, puzzle, i, j):
    temp = puzzle[i]
    puzzle[i] = puzzle[j]
    puzzle[j] = temp
    if random.randint(1, 3) == 1:
        cost = self.calculate_cost(puzzle)
    else:
        cost = self.get_less(puzzle)
    if not self.get_less(puzzle) % 2 == 0:
        print("************ NOT SOLVABLE **************")

    return (puzzle, cost)


def calculate_cost(self, puzzle):
    count = 0
    for i, j in zip(self.goal, puzzle):
        # print("i  j ",i , j)
        if i != j and j != 8:
            count += 1
            # print("count",count)
    return count


def get_less(self, puzzle):
    count = 0
    for i in range(0, len(puzzle)):
        for j in range(i, len(puzzle)):
            if puzzle[i] > puzzle[j] and puzzle[i] != 8:
                count += 1

    return count






