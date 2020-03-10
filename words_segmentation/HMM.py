class HMM(object):
    def __init__(self):
        # 主要是用于存取算法中间结果，不用每次都训练模型
        self.model_file = 'D:\\PROJECTS\\DATA\\HMM_CHN_SEG\\hmm_model.pkl'
        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        # 参数加载,用于判断是否需要重新加载model_file
        self.load_para = False

    # 用于加载已计算的中间结果，当需要重新训练时，需初始化清空结果
    def try_load_model(self, is_trained):
        if is_trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率（状态->状态的条件概率）
            self.A_dic = {}
            # 发射概率（状态->词语的条件概率）
            self.B_dic = {}
            # 状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    # 计算转移概率、发射概率以及初始概率
    def train(self, path):

        # 重置几个概率矩阵
        self.try_load_model(False)

        # 统计状态出现次数，求p(o)
        Count_dic = {}

        # 初始化参数
        def init_parameters():
            for state in self.state_list:  # 观测概率矩阵
                self.A_dic[state] = {s: 0.0 for s in self.state_list}  # 状态转移矩阵
                self.Pi_dic[state] = 0.0  # 初始概率矩阵
                self.B_dic[state] = {}

                Count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

            return out_text

        init_parameters()
        line_num = -1
        # 观察者集合，主要是字以及标点等
        words = set()
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                word_list = [i for i in line if i != ' ']
                words |= set(word_list)  # 求并集合，更新字的集合

                linelist = line.split()

                line_state = []  # ['B', 'S', 'BE', 'BME', 'S', ...]
                for w in linelist:
                    line_state.extend(makeLabel(w))

                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率 {'B': 0.0, 'S': 0.0}
                    else:
                        self.A_dic[line_state[k - 1]][v] += 1  # 计算转移概率
                        self.B_dic[line_state[k]][word_list[k]] = \
                            self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0  # 计算发射概率

        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.A_dic.items()}
        # 加1平滑
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.B_dic.items()}
        # 序列化
        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        ''' states, 状态序列。 start_p, trans_p, emit_p 初始状态矩阵、转移概率矩阵、发射概率矩阵 '''
        # step1, initialize
        V = [{}]  # 当前时刻、状体的概率
        path = {}  # 当前时刻、状体的路径
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        # V    [{'B': 0.0023862690310792406, 'M': 0.0, 'E': 0.0, 'S': 0.0008238177173969805}]
        # path {'B': ['B'], 'M': ['M'], 'E': ['E'], 'S': ['S']}

        # step2, recursion
        chars_all = set(emit_p['B'].keys()) | set(emit_p['M'].keys()) | set(emit_p['E'].keys()) | set(emit_p['S'].keys())
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in chars_all
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词
                (prob, state) = \
                    max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                new_path[y] = path[state] + [y]
            path = new_path
        # V, shape = len(text) x len(states)
        # path, shape = len(states) x len(text)

        # step3, terminate
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return prob, path[state]

    def cut(self, text):
        if not self.load_para:
            self.try_load_model(True)
        prob, positions = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = positions[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]

if __name__ == '__main__':
    hmm = HMM()
    #hmm.train('./data/data/trainCorpus.txt_utf8')

    text = '我爱中国 半年'
    res = hmm.cut(text)
    print(text)
    print(str(list(res)))













