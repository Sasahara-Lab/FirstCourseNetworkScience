'''
Simulation class for Chapter 7 Tutorial of Intro Network Science book

Copyright 2018 Indiana University and Cambridge University Press

※訳者注：このモジュールではすべてのコメントを日本語に訳していますが、
Pythonや他の言語では、ドキュメントのコメントを英語で書くことが
合意された規範ですので、オープンソースプロジェクトをリリースする場合は、
読者はこの点にご注意ください。
'''

from collections import Counter
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


class StopCondition(StopIteration):
    pass

class Simulation:
    '''ネットワークの状態遷移をシミュレートする。'''

    def __init__(self, G, initial_state, state_transition,
            stop_condition=None, name=''):
        '''
        シミュレーションのインスタンスを作成する。

        引数：
            G: networkx.Graph インスタンス。
            initial_state: `initial_state(G)`というシグネチャを持つ関数。
                グラフを引数として受け取り、すべてのノード状態の辞書を返す。
                この辞書のキーはノード名であり、値は対応するノードの初期状態である。
            state_transition: state_transition(G,current_state)
                というシグネチャを持つ関数。グラフと現在状態の引数
                を受け取り、更新されたノード状態の辞書を返す。この辞書のキーは
                ノード名で、値は対応する更新されたノード状態である。
            stop_condition（オプション）：stop_condition(G,current_state)`
                のシグネチャを持つ関数。
                グラフと現在のノード状態の辞書の2つの引数を受け取り、ブール値を返す。
                シミュレーションを現在の状態で停止させたい場合は真を返す。

        キーワード引数：
            name (オプション): プロットや図面のタイトルに使用される文字列。

        エラー：
            ValueError: 初期状態がないノードが存在する場合。
        '''
        self.G = G.copy()
        self._initial_state = initial_state
        self._state_transition = state_transition
        self._stop_condition = stop_condition
        # stop_condition=False と指定してもいい
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or 'Simulation'

        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')

        self._initialize()

        self._pos = nx.layout.spring_layout(G)

    def _append_state(self, state):
        self._states.append(state)
        # self._value_index を更新
        for value in set(state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)

    def _initialize(self):
        if self._initial_state:
            if callable(self._initial_state):
                state = self._initial_state(self.G)
            else:
                state = self._initial_state
            nx.set_node_attributes(self.G, state, 'state')

        if any(self.G.nodes[n].get('state') is None for n in self.G.nodes):
            raise ValueError('All nodes must have an initial state')

        self._append_state(state)

    def _step(self):
        # ノードの属性を真実のソースとして使うことにしている。
        # これにより、ユーザーはステップの合間に手動で
        # ネットワークに手を加えることができる。
        state = nx.get_node_attributes(self.G, 'state')
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        state = nx.get_node_attributes(self.G, 'state')
        new_state = self._state_transition(self.G, state)
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)

    def _categorical_color(self, value):
        index = self._value_index[value]
        node_color = self._cmap(index)
        return node_color

    @property
    def steps(self):
        '''シミュレーションが実行されたステップ数を返す。'''
        return len(self._states) - 1

    def state(self, step=-1):
        '''
        シミュレーションの状態を返す；指定されない場合は現在の状態を返す。

        引数：
            step: 返してもらいたい状態のステップ。デフォルトは -1 で、
            現在の状態を意味する。

        戻り値：
            ノードとその状態の辞書.

        エラー:
            IndexError: `step` 引数が実行されたステップ数より大きい場合。
        '''
        try:
            return self._states[step]
        except IndexError:
            raise IndexError('Simulation step %i out of range' % step)

    def draw(self, step=-1, labels=None, **kwargs):
        '''
        networkx.draw を使って、シミュレーションの状態をノードの状態値で
        色分けして描画する。デフォルトでは、現在の状態を描画する。

        引数：
            step: 描画するシミュレーションのステップ。デフォルトは -1 で、
            現在の状態を代表する。
            kwargs: networkx.draw() のキーワード引数。

        発生：
            IndexError: 引数 `step` がステップ数より大きい場合。
        '''
        state = self.state(step)
        node_colors = [self._categorical_color(state[n]) for n in self.G.nodes]
        nx.draw(self.G, pos=self._pos, node_color=node_colors, **kwargs)

        if labels is None:
            labels = sorted(set(state.values()), key=self._value_index.get)
        patches = [mpl.patches.Patch(color=self._categorical_color(l), label=l)
                   for l in labels]
        plt.legend(handles=patches)

        if step == -1:
            step = self.steps
        if step == 0:
            title = 'initial state'
        else:
            title = 'step %i' % (step)
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)

    def plot(self, min_step=None, max_step=None, labels=None, **kwargs):
        '''
        pyplot を使用して、ある範囲で（デフォルトは全体）
        各シミュレーションステップの各状態を持つノードの相対数をプロットする。

        引数
            min_step: 描画するシミュレーションの最初のステップ。
            デフォルトはNoneで、初期状態からプロットする。
            max_step: 描画するシミュレーションの最後のステップ。
            デフォルトはNoneで、現在のステップまでプロットする。
            labels: プロットする状態値の順序。
            デフォルトは観測された全ての状態値。
            kwargs: plt.plot() に渡されるキーワード引数。

        戻り値：
            現在のプロットの Axes オブジェクト。
        '''
        x_range = range(min_step or 0, max_step or len(self._states))
        counts = [Counter(s.values()) for s in self._states[min_step:max_step]]
        if labels is None:
            labels = {k for count in counts for k in count}
            labels = sorted(labels, key=self._value_index.get)

        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            plt.plot(x_range, series, label=label, **kwargs)

        title = 'node state proportions'
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)
        plt.xlabel('Simulation step')
        plt.ylabel('Proportion of nodes')
        plt.legend()
        plt.xlim(x_range.start)

        return plt.gca()

    def run(self, steps=1):
        '''
        引数 `steps` で指定した回数だけシミュレーションを実行する。
        デフォルトは 1 回である。

        引数：
            steps: シミュレーションを進めるステップ数。
        '''
        for _ in range(steps):
            try:
                self._step()
            except StopCondition as e:
                print(
                    "Stop condition met at step %i." % self.steps
                    )
                break

