import pickle
import os

class HMM:
    def __init__(self):
        self.A_dic = {}  # 转移概率矩阵
        self.B_dic = {}  # 观测状态矩阵
        self.Pi_dic = {}  # 初始状态矩阵
        self.state_list = ['B', 'M', 'E', 'S']  # 状态空间

    def train(self, data_path, save_path):
        # step1, init parameter
        for state in self.state_list:
            self.A_dic[state] = {s: 0.0 for s in self.state_list}  # 4x4
            self.B_dic[state] = {}
            self.Pi_dic[state] = 0.0

        # step 2, train Pi_dic, A_dic, B_dic
        line_num = 0  # 与源码不同
        chars_set = set()  # 字及标点等
        count_dic = {s: 0 for s in self.state_list}
        with open(data_path, encoding='UTF-8') as f:
            for line in f:
                # eg: line, '单纯 追求 声 、 色 、 光 、 影 或 无言 的 长 镜头 段落 ， '
                line_num += 1
                line = line.strip()
                if not line:
                    continue  # 空行

                chars = [i for i in line if i != ' ']  # 字
                chars_set |= set(chars)
                words = line.split()  # 词
                line_state = []  # eg, ['B', 'M', 'S', 'S', 'S', 'B', 'S']
                for word in words:
                    line_state.extend(self.label(word))
                assert len(chars) == len(line_state)

                for k, s in enumerate(line_state):
                    count_dic[s] += 1
                    if k == 0:
                        self.Pi_dic[s] += 1
                    else:
                        self.A_dic[line_state[k-1]][s] += 1
                        char = chars[k]
                        self.B_dic[line_state[k]][char] = self.B_dic[line_state[k]].get(char, 0) + 1.0
        self.Pi_dic = {m: n/line_num for m, n in self.Pi_dic.items()}
        self.A_dic = {m: {m1: n1/count_dic[m]for m1, n1, in n.items()} for m, n in self.A_dic.items()}
        self.B_dic = {m: {m1: n1/count_dic[m] for m1, n1 in n.items()}for m, n in self.B_dic.items()}

        # step3, save model
        self.save_model(save_path)

    def label(self, word):
        if len(word) == 1:
            res = ['S']
        else:
            res = ['B'] + ['M'] * (len(word)-2) + ['E']
        return res

    def save_model(self, path):
        if os.path.exists(path):
            print('model file is already existed')
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.A_dic, f)
                pickle.dump(self.B_dic, f)
                pickle.dump(self.Pi_dic, f)
            print('model finished training and saving')

    def words_seg(self, text, model_path):
        self.load_model(model_path)
        prob, states = self.viterbi(text)
        words = list(self.get_words(text, states))
        return prob, words

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.A_dic = pickle.load(f)
            self.B_dic = pickle.load(f)
            self.Pi_dic = pickle.load(f)

    def viterbi(self, text):
        # step1, initialize
        prob = [{}]  # 当前时刻、状态的概率
        path = {}  # 当前时刻、状态的路径
        for s in self.state_list:
            prob[0][s] = self.Pi_dic[s] * self.B_dic[s].get(text[0], 0)
            path[s] = [s]
        # step2, recursion
        chars_all = set(self.B_dic['B'].keys()) | set(self.B_dic['M'].keys()) | \
                    set(self.B_dic['E'].keys()) | set(self.B_dic['S'].keys())
        for t in range(1, len(text)):
            prob.append({})
            new_path = {}
            char = text[t]
            is_seen = char in chars_all  # 字是否出现过
            for s1 in self.state_list:  # ['B', 'M', 'E', 'S']
                emit_prob = self.B_dic[s1].get(char, 0) if is_seen else 1.0
                prob_state = []
                for s2 in self.state_list:  # ['B', 'M', 'E', 'S']
                    if prob[t-1][s2] > 0:
                        prob_state.append((prob[t-1][s2] * self.A_dic[s2].get(s1, 0) * emit_prob, s2))
                (prob_char, state_char) = max(prob_state)
                prob[t][s1] = prob_char
                new_path[s1] = path[state_char] + [s1]
            path = new_path
        # step3, terminate
        if self.B_dic['M'].get(text[-1], 0) > self.B_dic['S'].get(text[-1], 0):
            (prob_res, state_res) = max([(prob[len(text)-1][s], s) for s in ('E', 'M')])
        else:
            (prob_res, state_res) = max([(prob[len(text)-1][s], s) for s in self.state_list])
        return prob_res, path[state_res]

    def get_words(self, text, states):
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = states[i]
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
    data_path = 'D:\\PROJECTS\\DATA\\HMM_CHN_SEG\\trainCorpus.txt_utf8'
    model_path = 'D:\\PROJECTS\\DATA\\HMM_CHN_SEG\\hmm_ethan_model.pkl'
    # hmm.train(data_path, model_path)
    text = '我爱中国 半年'
    hmm.words_seg(text, model_path)
