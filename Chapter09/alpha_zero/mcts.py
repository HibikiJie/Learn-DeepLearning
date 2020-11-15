

class Node:

    def __init__(self, parent_node=None, p_hit=1.0):

        """设定父节点，方便查找"""
        self.parent_node = parent_node

        """创建空白子节点"""
        self.child_nodes = []

        """设定节点属性"""
        self.num_visits = 0  # 访问次数
        self.odds = 0  # 胜率
        self.hit_rate = 0  # 访问率
        self.p_hit = p_hit  # 访问的概率
        self.c_puct = 2

    def expand(self, a):
        """拓展子节点"""
        self.child_nodes.append(a)

    def select(self,):
        pass

    def backward(self,):
        pass

    def get_value(self):
        u = self.c_puct * self.p_hit * (self.parent_node.num_visits**0.5)/(1+self.num_visits)
        return self.odds + u


class MCSTree:

    def __init__(self):
        self.root = Node()



