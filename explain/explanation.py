import os

import networkx
import torch
import networkx as nx
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as pyg

from torch_geometric.data import Data
from typing import *

class Explanation(Data):
    """
    Save information of explainer subgraph.
    """
    plot_dict = {
        "SEED"       : 10,
        "node_dict"  : {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
                        7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
        # "node_colors": ("#E4A031, #D68438, #C76B60, #B55384, #7C4D77, #474769, #26445E, "
        #                 "#4C7780, #73A5A2, #F6E2C1, #F3DBC1, #B2B6C1, #D6E2E2, #F0EFED").split(', '),
        "node_colors": ['#E49D1C', '#4970C6', '#73A5A2', '#FF5357',
                        'green',  'darkslategray', '#F0EA00'],
        "edge_colors": ['gray'],
        "motif_colors": ['red', 'lime']
    }

    show_fig = False

    def plot_subgraph(
        self,
        x: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_index  : torch.Tensor,
        topk_edges  : int = 4,
        display_mode: str = "highlight",
        save_name   : Optional[str] = None,
        save_path   : Optional[str] = None,
        title_sentence: Optional[str] = None,
        is_original: bool = False
    ):
        r"""
        Visualize explanation subgraph based on networkx.

        Args:
            x: node features of graph
            edge_weights: edge mask generated from explainer
            edge_index  : adjacency matrix of graph
            topk_edges  : the size of subgraph
            node_size   : the size of node in figure
            edge_size   : the size of edges in figure
            save_name   : the name of figure
            save_path   : the save path of figure
            display_mode : which way to display the subgraph, the optional includes ["bold", "highlight"]
        """
        # node labels
        node_indices = x.sum(dim=-1).view(-1)
        # clear noise nodes
        x = x[node_indices != 0]
        node_labels  = x.argmax(-1).tolist()
        # remove self loops
        edge_indices = edge_index[0] != edge_index[1]
        edge_index   = edge_index[:, edge_indices]
        # get topk edges
        topk_values, topk_indices = torch.topk(edge_weights[edge_indices], topk_edges)
        # get the clear graph
        pyg_graph: Data    = Data(x=x, edge_index=edge_index)
        # get networkx graph
        nx_graph: nx.Graph = pyg.utils.to_networkx(pyg_graph, to_undirected=True)
        # get explanation subgraph
        expl_edge_index = edge_index[:, topk_indices]
        expl_subgraph   = nx.Graph()
        expl_subgraph.add_nodes_from(expl_edge_index.flatten().unique().tolist())
        expl_subgraph.add_edges_from(expl_edge_index.T.tolist())
        # nodes and edges in explanations
        node_list = [list(cpn) for cpn in nx.connected_components(expl_subgraph)]
        edge_list = list(expl_subgraph.edges())

        # get colors of nodes
        colors = [self.plot_dict["node_colors"][v % len(self.plot_dict["node_colors"])]
                  for k, v in enumerate(node_labels)]
        if display_mode == "highlight":
            self._display_in_highlight(
                nx_graph,
                node_labels      = {idx: self.plot_dict["node_dict"][val] for idx, val in enumerate(node_labels)},
                node_colors      = colors,
                expl_node_list   = node_list,
                expl_edge_list   = edge_list,
                save_name        = save_name,
                save_path        = save_path,
                is_original      = is_original
            )
        else:
            self._display_in_bold(
                nx_graph,
                # node_labels   ={idx: idx for idx, _ in enumerate(nx_graph.nodes())},
                node_labels={idx: self.plot_dict["node_dict"][val] for idx, val in enumerate(node_labels)},
                node_colors   = colors,
                expl_node_list=node_list,
                expl_edge_list=edge_list,
                save_name=save_name,
                save_path=save_path,
                is_original=is_original
            )

    def _display_in_highlight(
        self,
        ori_graph: nx.Graph,
        node_labels   : Dict[int, Union[int, str]],
        node_colors   : List[str],
        expl_node_list: List[int],
        expl_edge_list: List[Tuple[int, int]],
        save_name     : str,
        save_path     : str,
        **kwargs
    ):
        pos = nx.kamada_kawai_layout(ori_graph)

        # draw explanation subgraph nodes
        for i, cluster in enumerate(expl_node_list):
            pos_node_list = {k: v for k, v in pos.items() if k in cluster}
            nx.draw_networkx_nodes(
                ori_graph,
                pos_node_list,
                nodelist=cluster,
                node_color=self.plot_dict["motif_colors"][-1],
                alpha=0.5,
                node_shape="o",
                node_size=kwargs.get('motif_node_size', 300),
            )

        # draw original graph nodes
        nx.draw_networkx_nodes(
            ori_graph,
            pos,
            nodelist=list(ori_graph.nodes()),
            node_color=node_colors,
            node_size=kwargs.get('node_size', 100)
        )

        # draw original graph edges
        nx.draw_networkx_edges(
            ori_graph,
            pos,
            width=3,
            edge_color=self.plot_dict['edge_colors'][-1],
            arrows=False
        )

        # draw explanations subgraph edges
        nx.draw_networkx_edges(
            ori_graph,
            pos,
            width=8,
            edgelist=expl_edge_list,
            edge_color=self.plot_dict["motif_colors"][-1],
            alpha=0.5,
            arrows=False
        )
        if node_labels is not None:
            nx.draw_networkx_labels(ori_graph, pos, node_labels, font_weight='bold', font_size=8)

        plt.axis("off")

        # show figure
        if self.show_fig:
            plt.show()

        if save_path and save_name:
            # save figure
            file_name = osp.join(save_path, save_name)
            plt.savefig(file_name, dpi=200, bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Explanatory subgraph saving at path: {file_name}")

    def _display_in_bold(
        self,
        ori_graph: nx.Graph,
        node_labels   : Dict[int, Union[int, str]],
        node_colors   : List[int],
        expl_node_list: List[int],
        expl_edge_list: List[Tuple[int, int]],
        save_name: str,
        save_path: str,
        **kwargs
    ):

        pos = nx.kamada_kawai_layout(ori_graph)
        # pos = nx.shell_layout(ori_graph)
        # pos = nx.spectral_layout(ori_graph)
        # pos = nx.spiral_layout(ori_graph)
        # offset = np.array([0.5, 0.5])
        for i in range(1, 6):
            pos[i] += i / 20
        for i in range(21):
            if i not in range(1, 6):
                pos[i] += 1 / 20
        pos[21] += 1 / 10
        pos[10] += 1 / 20
        pos[11] += 1 / 20
        pos[12] += 1 / 20
        # draw original graph nodes
        nodes = nx.draw_networkx_nodes(
            ori_graph,
            pos,
            nodelist=list(ori_graph.nodes()),
            node_color=node_colors,
            node_size=kwargs.get('node_size', 300),
            edgecolors='black',
            linewidths=1.5
        )

        if not (is_original := kwargs.get('is_original', False)):
            nodes.set_edgecolor('black')
            nodes.set_linestyle('--')

        # draw original graph edges
        nx.draw_networkx_edges(
            ori_graph, pos, width=3, edge_color="gray", arrows=False, style='dashed' if not is_original else 'solid'
        )

        if not is_original:
            for i, cluster in enumerate(expl_node_list):
                pos_node_list = {k: v for k, v in pos.items() if k in cluster}
                node_colors_  = [node_colors[k] for k in cluster]
                nx.draw_networkx_nodes(
                    ori_graph,
                    pos_node_list,
                    nodelist=cluster,
                    node_color=node_colors_,
                    node_size=kwargs.get('node_size', 300),
                    edgecolors='black',
                    linewidths=2.5
                )

            # draw explanations subgraph edges
            nx.draw_networkx_edges(
                ori_graph,
                pos,
                width=5,
                edgelist=expl_edge_list,
                edge_color="black",
                arrows=False
            )
        if node_labels is not None:
            nx.draw_networkx_labels(ori_graph, pos, node_labels, font_weight='bold', font_size=8)

        plt.axis("off")

        # show figure
        if self.show_fig:
            plt.show()

        if save_path and save_name:
            # save figure
            os.makedirs(save_path, exist_ok=True)
            file_name = osp.join(save_path, save_name)
            plt.savefig(file_name, dpi=200, bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Explanatory subgraph saving at path: {file_name}")

    def plot_subgraph_old(
        self,
        x: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_index  : torch.Tensor,
        topk_edges  : int = 4,
        node_size   : int = 200,
        edge_size   : int = 3,
        save_name   : Optional[str] = None,
        save_path   : Optional[str] = None,
        display_mode: str = "bold"
    ):
        r"""
        Visualize explanation subgraph based on networkx.

        Args:
            x: node features of graph
            edge_weights: edge mask generated from explainer
            edge_index  : adjacency matrix of graph
            topk_edges  : the size of subgraph
            node_size   : the size of node in figure
            edge_size   : the size of edges in figure
            save_name   : the name of figure
            save_path   : the save path of figure
            display_mode : which way to display the subgraph, the optional includes ["bold", "highlight"]
        """
        # node labels
        node_indices = x.sum(dim=-1).view(-1)
        # clear noise nodes
        x = x[node_indices != 0]
        node_labels  = x.argmax(-1).tolist()
        # remove self loops
        edge_indices = edge_index[0] != edge_index[1]
        edge_index   = edge_index[:, edge_indices]
        # transform to networkx graph
        G   = pyg.utils.to_networkx(Data(x=x, edge_index=edge_index), to_undirected=True)
        # graph layout
        # pos = nx.circular_layout(G)
        # pos = nx.bipartite_layout(G)
        if display_mode == 'bold':
            self._display_in_bold(G)
        else:
            self._display_in_highlight(G)
        pos = nx.kamada_kawai_layout(G)

        # setting the attribute of nodes and edges
        node_colors = [self.node_colors[node_label] for node_label in node_labels]
        node_sizes  = [node_size] * G.number_of_nodes()
        edge_colors = [self.edge_colors[0]] * G.number_of_edges()
        edge_width  = [edge_size] * G.number_of_edges()

        # find topk edges
        topk_value, topk_indices = torch.topk(edge_weights[edge_indices], topk_edges)

        # highlight nodes
        highlight_nodes = edge_index[:, topk_indices].flatten().unique().tolist()

        # highlight edges
        highlight_edges_dict = set()
        for u, v in edge_index[:, topk_indices].T.tolist():
            if (u, v) not in highlight_edges_dict and (v, u) not in highlight_edges_dict:
                highlight_edges_dict.add((u, v))
        highlight_edges = list(highlight_edges_dict)

        # draw original graph
        plt.figure(figsize=(4, 3))
        nx.draw(G,
                pos,
                with_labels=False,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color=edge_colors,
                width=edge_width)

        # setting highlight attribute
        highlight_color = 'green'
        highlight_alpha = 0.5
        highlight_size_multiplier = 3
        highlight_width_multiplier = 5
        highlight_node_size  = node_size * highlight_size_multiplier
        highlight_edge_width = edge_width * highlight_width_multiplier

        # draw highlight nodes
        for node in highlight_nodes:
            x, y = pos[node]
            plt.scatter([x], [y],
                        s=highlight_node_size,
                        edgecolors='none',
                        facecolors=highlight_color,
                        alpha=highlight_alpha)

        # draw highlight edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=highlight_edges,
            edge_color=highlight_color,
            width=highlight_edge_width,
            alpha=highlight_alpha
        )

        # show figure
        plt.show()

        if save_path and save_name:
            # save figure
            file_name = osp.join(save_path, save_name)
            plt.savefig(file_name)
            print(f"Explanatory subgraph saving at path: {file_name}")

    def plot_explanation_subgraph(
        self,
        topk_edges: int = 4,
        save_name: Optional[str] = None,
        save_path: Optional[str] = None,
        display_mode: str = 'bold',
        is_original: bool = False
    ):
        r"""
        visualize explanation subgraph.
        """
        self.plot_subgraph(self.x,
                           self.edge_mask,
                           self.edge_index,
                           topk_edges,
                           display_mode,
                           save_name,
                           save_path,
                           title_sentence=None,
                           is_original=is_original)

    def plot_ground_truth_subgraph(
        self,
        save_name: Optional[str] = None,
        save_path: Optional[str] = None,
        display_mode: str = 'bold'
    ):
        r"""
        Visualize ground truth subgraph.
        """
        assert self.ground_truth_mask is not None
        topk_edges = int((self.ground_truth_mask == 1).sum(-1))
        self.plot_subgraph(self.x,
                           self.ground_truth_mask,
                           self.edge_index,
                           topk_edges,
                           display_mode,
                           save_name,
                           save_path,
                           title_sentence=None)