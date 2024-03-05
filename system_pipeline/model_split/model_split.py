import warnings
import torch
import torch.nn as nn
import torch.fx
from transformers.utils.fx import symbolic_trace, HFTracer
from torch.fx import Graph, Node
from torch.distributed.rpc import RRef
from collections import OrderedDict
from util.module_utils import add_method
from torch.fx import GraphModule
import transformers


class ModelSplit:
    def __init__(self, m: nn.Module, debug_mod: bool = False, model_allocation=[], split_option="fixed"):
        self.mod = m
        self.mod_mapping = {}
        self.subgraph_dependency_bool = True
        self.device_range = []
        self.debug_mod = debug_mod
        self.model_allocation = model_allocation
        self.split_option = split_option
        self.max_split_size = 0

    def module_split_rref(self, subgraphs: torch.fx.GraphModule.graph):
        # TODO: will be implemented in the distributed training part.
        pass

    def graph_module_split(self, gm: torch.fx.GraphModule, split_size: int) -> list:
        """
            Method for splitting a module graph based on split_size
        """

        graph = gm.graph
        node_list = [i for i in gm.graph.nodes]
        subgraph_list = []

        # find call_module nodes
        call_module_idx = [node_list.index(i) for i in node_list if i.op == "call_module"]

        # if the second last node's op is call_module, we update it's node idx in call_module_idx by 1.
        if node_list[call_module_idx[-1] + 1].op == "output":
            call_module_idx[-1] = call_module_idx[-1] + 1

        # Check whether dependent subgraphs exist and create available device range for splitting
        self.find_subgraph_user_based(graph)

        # split based on node dependency
        if self.subgraph_dependency_bool:
            # default split based on the even model partition given the split size
            temp_split_idx = self._subgraph_layer_combination(graph, split_size, split_option=self.split_option)
            split_idx = self.subgraph_node_check(graph, temp_split_idx)

        self.max_split_size = len(split_idx)

        if not self.subgraph_dependency_bool:
            self.device_range = [1, len(call_module_idx)]

        if split_size in range(self.device_range[0], self.device_range[1] + 1):
            pass
        else:
            raise RuntimeError(f"Number of split size {split_size} exceeds the maximum number of required devices."
                               f"Redefine the split size in range {self.device_range}.")

        # Start creating splitted module graphs
        for idx_list in split_idx:
            start_idx = idx_list[0]
            end_idx = idx_list[-1]
            subnodes = node_list[start_idx:end_idx]
            # print(subnodes)
            # create new graph
            subgraph = Graph()
            node_names = []
            value_remap = {}
            for node in subnodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.name not in node_names:
                        value_remap[arg] = subgraph.placeholder(arg.name)
                        node_names.append(arg.name)
                node_names.append(node.name)

            # mod stands for nn.Module
            for node in subnodes:
                value_remap[node] = subgraph.node_copy(node, lambda n: value_remap[n])
                if node.op == "call_module":
                    mod = self.get_module(node.target)
                    self.mod_mapping[node.target] = mod

            # retreive all previous subgraphs when subgraph_list is not empty
            subgraph_last_node = [node for node in subgraph.nodes][-1]

            # get subgraph last node as graph output
            subgraph.create_node(op='output', target='output', name='output', args=tuple([subgraph_last_node]))
            subgraph_list.append(subgraph)

        self.subgraphs_initial = subgraph_list
        return subgraph_list

    def _create_multiple_return_sequential(self, subgraph_list_org: list):
        # backward search
        subgraph_list = subgraph_list_org.copy()
        self.mod_mapping = {}
        out = []

        for graph_index in reversed(range(len(subgraph_list))):
            # get placeholder names in the last subgraph
            current_subgraph = subgraph_list[graph_index]

            placeholder_node_current = [(node, node.name) for node in current_subgraph.nodes
                                        if node.op == "placeholder"]
            placeholder_node_current_copy = placeholder_node_current.copy()
            output_searched_node = {}

            # search previous graph
            search_index = graph_index - 1
            subgraph_to_search = subgraph_list[search_index]

            # check placeholder nodes in subgraph_to_search
            placeholder_node_to_search = [i for i in subgraph_to_search.nodes if i.op == 'placeholder']
            placeholder_node_name_to_search = [i.name for i in subgraph_to_search.nodes if i.op == 'placeholder']
            nodes_in_to_search = [i for i in subgraph_to_search.nodes]
            node_name_in_to_search = [i.name for i in subgraph_to_search.nodes]

            if search_index >= 0:
                for node, node_name in placeholder_node_current:
                    # check whether last graph's placeholder exist in previous graph
                    past_subgraph_nodes = [(node, node.name) for node in subgraph_to_search.nodes]
                    for pre_node, pre_node_name in past_subgraph_nodes[::-1]:
                        # if current placeholder name existed in previous graph, we add this node to the searched_node
                        # then we remove it from current search list - the placeholder_node_current.

                        if node_name == pre_node_name:
                            output_searched_node[pre_node_name] = pre_node

                # after the for loop, we check whether or not the placeholder_node_current is empty
                # if it's not empty, we add the remaining node to the previous graph as placeholder and output
                # then go to the lower numbered subgraph
                for remain_node, remain_node_name in placeholder_node_current:
                    # add the node to the output_searched_node dictionary
                    if remain_node_name not in placeholder_node_name_to_search and \
                            remain_node_name not in node_name_in_to_search:
                        node_before_insertion = placeholder_node_to_search[-1]
                        with subgraph_to_search.inserting_after(node_before_insertion):
                            new_node = subgraph_to_search.placeholder(remain_node_name)
                            output_searched_node[remain_node_name] = new_node

                # reorder the output searched node dict
                output_searched_node = OrderedDict(
                    (v, output_searched_node[v]) for k, v in placeholder_node_current_copy)

                out_nodes = [val for key, val in output_searched_node.items()]

                for node in subgraph_to_search.nodes:
                    if node.name == "output":
                        subgraph_to_search.erase_node(node)
                        subgraph_to_search.output(out_nodes)
            else:
                break
        for new_graph in subgraph_list:
            subgraph = Graph()
            node_names = []
            value_remap = {}

            for node in new_graph.nodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.name not in node_names:
                        value_remap[arg] = subgraph.placeholder(arg.name)
                        node_names.append(arg.name)

                node_names.append(node.name)

            # mod stands for nn.Module
            for node in new_graph.nodes:
                value_remap[node] = subgraph.node_copy(node, lambda n: value_remap[n])
                if node.op == "call_module":
                    mod = self.get_module(node.target)
                    self.mod_mapping[node.target] = mod
                elif node.op == "get_attr":
                    # Split the path and traverse the hierarchy
                    attr_path = node.target.split('.')
                    attr = self.mod
                    for p in attr_path:
                        if p.isdigit():  # Accessing a list or tuple by index
                            attr = attr[int(p)]
                        else:  # Accessing an attribute by name
                            attr = getattr(attr, p)
                    self.mod_mapping[node.target] = attr
            out.append(subgraph)

        return out

    def _create_multiple_return_residual(self, subgraph_list_org: list):
        # start manipulating the placeholder.
        # we assume the first submodule does not require input modification.
        # therefore, the submodules after the submodule0 needs placeholder manipulation.
        subgraphs = subgraph_list_org.copy()
        self.mod_mapping = {}
        out = []

        # retrieve the sequential and residual dependency maps for graph manipulation
        sequential_dependency_map, residual_dependency_map = self.residual_connection_search(subgraphs)
        sequential_dependency_flatten, residual_dependency_flatten = {}, {}

        # data structure for maintaining the placeholder node names in sequential and residual for each submodule
        # for later graph manipulation purpose
        # example for submodule_input_order_map:
        """
        {1: {'sequential': ['transformer_h_4_post_attention_layernorm', 'add_18', 'to', 'or_'], 'residual': []}, 
        2: {'sequential': ['transformer_h_9_post_attention_layernorm', 'add_38'], 'residual': [to, or_]}, 
        3: {'sequential': ['transformer_h_14_post_attention_layernorm', 'add_58'], 'residual': [to, or_]}, 
        4: {'sequential': ['transformer_h_19_post_attention_layernorm', 'add_78'], 'residual': [to, or_]}}
        """
        submodule_input_order_map = {}

        for sequential_graph_index in range(len(subgraphs) - 1):
            sequantial_graph_output = [node for node in
                                       sequential_dependency_map[sequential_graph_index][sequential_graph_index + 1]]
            submodule_input_order_map[sequential_graph_index + 1] = {"sequential": sequantial_graph_output,
                                                                     "residual": []}

        for residual_graph_index in range(len(subgraphs)):
            # residual dependency map example:  {0: {4: [to, or_], 3: [to, or_], 2: [to, or_]}}
            if residual_graph_index in residual_dependency_map:
                # residual model node map example: {4: [to, or_], 3: [to, or_], 2: [to, or_]}
                residual_model_node_map = residual_dependency_map[residual_graph_index]
                for i in range(residual_graph_index, len(subgraphs)):
                    if i in residual_model_node_map:
                        submodule_input_order_map[i]["residual"] += [node for node in residual_model_node_map[i]]

        # with the submodule_input_order_map, we need to reorder the placeholder arguments' order
        # for each model it should concatenate the sequential and residual model outputs input a single array
        # for further inference proceeding
        for graph_index in range(1, len(subgraphs)):
            new_graph = torch.fx.Graph()
            current_graph = subgraphs[graph_index]

            # reordered placeholder nodes based on submodule_input_order_map
            sequential_placeholder_names = submodule_input_order_map[graph_index]["sequential"]
            residual_placeholder_names = submodule_input_order_map[graph_index]["residual"]
            residual_placeholder_nodes = []
            for node in current_graph.nodes:
                if node.name in sequential_placeholder_names:
                    continue
                elif node.name in residual_placeholder_names:
                    residual_placeholder_nodes.append(node)

            # add residual nodes
            current_graph_residual_node_name = [i.name for i in residual_placeholder_nodes]
            residual_node_org_index = [current_graph_residual_node_name.index(i) for i in residual_placeholder_names]
            residual_node_reorder = [residual_placeholder_nodes[i] for i in residual_node_org_index]
            sequential_nodes = self.get_placeholder_nodes(current_graph)
            for residual_node in residual_node_reorder:
                last_placeholder_node = sequential_nodes[-1]
                with current_graph.inserting_after(last_placeholder_node):
                    new_node = current_graph.placeholder(residual_node.name)
                    residual_node.replace_all_uses_with(new_node)
                    for node in current_graph.nodes:
                        if node.name == residual_node.name:
                            current_graph.erase_node(node)
                    new_node.name = residual_node.name
                sequential_nodes.append(new_node)

        # now, placeholder processing is finished. We need to post-processing the output node
        # For each model, it needs to provide 2 outputs [sequential], [residual].
        # we need to first process how the residual output should be for the sender submodule.
        # Note: for the last submodule, we do not need to process its output since there would not be
        # residual connection exists.
        submodule_output_tensor_map = {}
        for graph_index in range(len(subgraphs) - 1):
            sequential_output_tensor = [node for node in sequential_dependency_map[graph_index][graph_index + 1]]
            submodule_output_tensor_map[graph_index] = {"sequential": sequential_output_tensor, "residual": []}

        for graph_index in range(len(subgraphs)):
            if graph_index in residual_dependency_map:
                seen = set()
                residual_output_tensors = []
                # keep the order of original residual dependency map as the same
                for key, val in residual_dependency_map[graph_index].items():
                    for item in val:
                        if item not in seen:
                            seen.add(item)
                            residual_output_tensors.append(item)
                submodule_output_tensor_map[graph_index]["residual"] += [node for node in residual_output_tensors]

        # we need to further manipulate each submodule's output node
        for graph_index in range(len(subgraphs)):
            if graph_index in submodule_output_tensor_map:
                sequential_node_names = submodule_output_tensor_map[graph_index]['sequential']
                residual_node_names = submodule_output_tensor_map[graph_index]['residual']
                current_graph = subgraphs[graph_index]
                all_node_current_graph = self.get_all_graph_nodes(current_graph)
                all_node_name_current_graph = self.get_all_graph_nodes_name(current_graph)
                sequential_output_nodes = [all_node_current_graph[i] for i in
                                           [all_node_name_current_graph.index(name) for name in sequential_node_names]]
                residual_output_nodes = [all_node_current_graph[i] for i in
                                         [all_node_name_current_graph.index(name)
                                          for name in residual_node_names]] if residual_node_names else []

                # flatten the output nodes since onnx must require flatten output
                temp_out_nodes = sequential_output_nodes + residual_output_nodes
                out_nodes = []
                for node in temp_out_nodes:
                    if node not in out_nodes:
                        out_nodes.append(node)
                out_nodes_name = [node.name for node in out_nodes]

                # create additional indexing map for sequential and residual dependency
                sequential_dependency_flatten[graph_index] = {
                    graph_index + 1: [out_nodes.index(node) for node in sequential_output_nodes]}
                if graph_index in residual_dependency_map:
                    residual_dependency_flatten[graph_index] = {}
                    for key, val in residual_dependency_map[graph_index].items():
                        residual_dependency_flatten[graph_index][key] = [out_nodes_name.index(node)
                                                                         for node in
                                                                         residual_dependency_map[graph_index][key]]

                for node in current_graph.nodes:
                    if node.name == "output":
                        current_graph.erase_node(node)
                        current_graph.output(out_nodes)

        # debugging log
        if self.debug_mod:
            subgraph_counter = 0
            for subgraph in subgraphs:
                print(f"Input and Output nodes for submodule {subgraph_counter}:")
                print(self.get_placeholder_nodes(subgraph))
                print(self.get_output_nodes(subgraph)[0].args)
                print("******************************\n")
                subgraph_counter += 1
            print("-" * 20)

        # reconstruct graphs
        for new_graph in subgraphs:
            subgraph = Graph()
            node_names = []
            value_remap = {}

            for node in new_graph.nodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.name not in node_names:
                        value_remap[arg] = subgraph.placeholder(arg.name)
                        node_names.append(arg.name)

                node_names.append(node.name)
            # mod stands for nn.Module
            for node in new_graph.nodes:
                value_remap[node] = subgraph.node_copy(node, lambda n: value_remap[n])
                if node.op == "call_module":
                    mod = self.get_module(node.target)
                    self.mod_mapping[node.target] = mod
                elif node.op == "get_attr":
                    # Split the path and traverse the hierarchy
                    attr_path = node.target.split('.')
                    attr = self.mod
                    for p in attr_path:
                        if p.isdigit():  # Accessing a list or tuple by index
                            attr = attr[int(p)]
                        else:  # Accessing an attribute by name
                            attr = getattr(attr, p)
                    self.mod_mapping[node.target] = attr
            out.append(subgraph)

        if self.debug_mod:
            print(f"\nSequential Dependency Map: {sequential_dependency_map}")
            print(f"\nResidual Dependency Map: {residual_dependency_map}\n")
            print(f"\nSequential Dependency flatten Index Map: {sequential_dependency_flatten}")
            print(f"\nResidual Dependency flatten Index Map: {residual_dependency_flatten}\n")
        return out, sequential_dependency_flatten, residual_dependency_flatten

    def residual_connection_search(self, subgraph_list: list):
        subgraphs = subgraph_list.copy()
        residual_dependency_map = {}
        sequential_dependency_map = {}
        output_node_to_remove = {}
        search_step = 0

        # start backward search
        # Residual -> {graph_index: {graph_index_n:[node]}}
        # Sequential -> {graph_index: {graph_index_n: [node]}}
        for graph_index in reversed(range(len(subgraph_list))):
            if graph_index == 0:
                break
            current_subgraph = subgraphs[graph_index]

            # get current subgraphs' placeholder nodes
            placeholder_nodes_name = self.get_placeholder_nodes_name(current_subgraph)

            # search all previous graphs before current graph index
            for prev_graph_index in reversed(range(0, graph_index)):
                # first check immeidate dependency
                # rule: not in output_node, in graph node, not coexist in placeholder nodes
                if prev_graph_index == graph_index - 1:
                    immidiate_subgraph = subgraphs[prev_graph_index]
                    # get immidiate subgraph names
                    immidiate_subgraph_output_nodes_name = self.get_output_nodes_name(immidiate_subgraph)
                    immidiate_subgraph_input_nodes_name = self.get_placeholder_nodes_name(immidiate_subgraph)
                    immidiate_subgraph_all_node_name = self.get_all_graph_nodes_name(immidiate_subgraph)

                    # get all immidiate graph node
                    immidate_subgraph_all_nodes = self.get_all_graph_nodes(immidiate_subgraph)

                    # build sequential denpendency map for message passing
                    sequential_dependency_map[prev_graph_index] = {graph_index: []}

                    for current_subgraph_p_name in placeholder_nodes_name:
                        if current_subgraph_p_name not in immidiate_subgraph_output_nodes_name and \
                                current_subgraph_p_name in immidiate_subgraph_all_node_name and \
                                current_subgraph_p_name not in immidiate_subgraph_input_nodes_name:
                            # retrieve the node index in immidiate prev subgraph
                            sequential_dependent_node_index = immidiate_subgraph_all_node_name.index(
                                current_subgraph_p_name)

                            # add the node to the sequential dependency map
                            sequential_dependency_map[prev_graph_index][graph_index].append(
                                immidate_subgraph_all_nodes[sequential_dependent_node_index].name
                            )
                # search residual node dependency. We search each previous graph until we find the first subgraph
                # that contains the nodes we need in the current subgraph's placeholder
                # rule: not in output node, in graph node, not coexist in placeholder node
                else:
                    # retrieve previous graph
                    prev_graph = subgraphs[prev_graph_index]
                    # retrieve previous graph node names
                    prev_graph_placeholder_nodes_name = self.get_placeholder_nodes_name(prev_graph)
                    prev_graph_output_nodes_name = self.get_output_nodes_name(prev_graph)
                    prev_graph_all_nodes_name = self.get_all_graph_nodes_name(prev_graph)

                    # retrieve previous graph node
                    prev_graph_nodes = self.get_all_graph_nodes(prev_graph)

                    # Initiate a list for the current prev_graph_index if not already initialized
                    if prev_graph_index not in residual_dependency_map:
                        residual_dependency_map[prev_graph_index] = {}

                    # If there's no entry for the current graph_index in the residual_dependency_map[prev_graph_index], initiate it
                    if graph_index not in residual_dependency_map[prev_graph_index]:
                        residual_dependency_map[prev_graph_index][graph_index] = []

                    for current_subgraph_p_name in placeholder_nodes_name:
                        if prev_graph_index != 0:
                            if current_subgraph_p_name not in prev_graph_output_nodes_name and \
                                    current_subgraph_p_name in prev_graph_all_nodes_name and \
                                    current_subgraph_p_name not in prev_graph_placeholder_nodes_name:
                                node_index = prev_graph_all_nodes_name.index(
                                    current_subgraph_p_name
                                )
                                residual_dependency_map[prev_graph_index][graph_index].append(
                                    prev_graph_nodes[node_index].name
                                )
                        else:
                            # edge case: residual dependency exist in first subgraph's placeholder.
                            if current_subgraph_p_name not in prev_graph_output_nodes_name and \
                                    current_subgraph_p_name in prev_graph_all_nodes_name or \
                                    current_subgraph_p_name in prev_graph_placeholder_nodes_name:
                                node_index = prev_graph_all_nodes_name.index(
                                    current_subgraph_p_name
                                )
                                residual_dependency_map[prev_graph_index][graph_index].append(
                                    prev_graph_nodes[node_index].name
                                )
        # post processing the residual dependency map
        # Remove inner keys with empty lists
        cleaned_data = {k: {inner_k: v for inner_k, v in inner_dict.items() if v} for k, inner_dict in
                        residual_dependency_map.items()}

        # Remove outer keys with empty inner dictionaries
        residual_dependency_map = {k: v for k, v in cleaned_data.items() if v}
        return sequential_dependency_map, residual_dependency_map

    def _split_list(self, list_in: list, split_size: int, split_option: str = "fixed") -> list:
        if split_option == "fixed":
            if len(list_in) <= split_size:
                raise Exception("split_size must smaller than the number of layers. "
                                "Double check the split_size value and the number layers in the model.")
            else:
                sublist_size = len(list_in) // split_size  # integer division to get sublist size
                remainder = len(list_in) % split_size  # get any remaining elements after splitting into sublists

                sublists = []
                start = 0

                for i in range(split_size):
                    end = start + sublist_size

                    if remainder > 0:  # if there are remaining elements, add one to the sublist size
                        end += 1
                        remainder -= 1
                    if sublists == []:
                        sublists.append(list_in[start:end])
                    else:
                        sublists.append(list_in[start - 1:end])
                    start = end  # move the start index to the end of the current sublist

                return sublists

    def get_subgraphs(self, split_size: int, remote_option: bool = True) -> list:
        """
            return a list of subgraphs with module maps -> [[module_map:{}, subgraph1:torch.fx.GraphModule.graph],
                                                            [module_map:{}, subgraph2:torch.fx.GraphModule.graph],.....]
        """
        if remote_option:
            gm: torch.fx.GraphModule = torch.fx.symbolic_trace(self.mod)
            subgraphs = self.graph_module_split(gm, split_size)
            for g in subgraphs:
                # test graph
                g.lint()
            subgraphs = self.module_split_rref(subgraphs)
            return [[self.mod_mapping, subgraphs[i]] for i in range(len(subgraphs))]
            # return subgraphs, self.mod_mapping
        else:
            gm: torch.fx.GraphModule = torch.fx.symbolic_trace(self.mod)
            subgraphs = self.graph_module_split(gm, split_size)
            for g in subgraphs:
                g.lint()
            return [[self.mod_mapping, subgraphs[i]] for i in range(len(subgraphs))]

            # return subgraphs, self.mod_mapping

    def split_module(self, split_size: int, remote_option=False, transformer_model_option=True,
                     residual_connection=True):
        # TODO: remote_option is for training part. Currently is set to False. module_split_rref is waiting to be
        #  implemented.
        sequential_dependency_map, residual_dependency_map = {}, {}

        if transformer_model_option:
            gm = transformers.utils.fx.symbolic_trace(self.mod)
            subgraphs = self.graph_module_split(gm, split_size)
            if residual_connection:
                subgraphs, sequential_dependency_map, residual_dependency_map = self._create_multiple_return_residual(
                    subgraphs)
            else:
                subgraphs = self._create_multiple_return_sequential(subgraphs)
        else:
            gm: torch.fx.GraphModule = torch.fx.symbolic_trace(self.mod)
            subgraphs = self.graph_module_split(gm, split_size)

        if remote_option:
            subgraphs = self.module_split_rref(subgraphs)

        gm_list = []
        for subgraph in subgraphs:
            new_gm = GraphModule(self.mod_mapping, subgraph)
            if remote_option:
                @add_method(GraphModule)
                def parameter_rrefs():
                    # when this method is called, it will return a list of module parameters which is saved as RRef object
                    return [RRef(p) for p in new_gm.parameters()]
            gm_list.append(new_gm)
        out_module = []
        for g in gm_list:
            # test subgraph where g is torch.fx.GraphModule
            g.graph.lint()

            # if there is no error
            # add the graph module to the output
            g.recompile()

            # assign device for splitted models. Default:
            if transformer_model_option:
                g.class_for_deserialization = self.mod.__class__
                g.config = self.mod.config
                g.device = self.mod.device
            out_module.append(g)

        if residual_connection:
            return out_module, sequential_dependency_map, residual_dependency_map
        else:
            return out_module

    def get_module(self, name):
        # self.mod is the nn.Module
        return dict(self.mod.named_modules())[name]

    def find_subgraph_user_based(self, graph: torch.fx.GraphModule.graph) -> OrderedDict:
        """
        Method for finding the subgraphs if starting at any graph node, the length of node.users is greater than 1
        At this point, there is a subgraph exists

        Input: torch.fx.GraphModule.graph
        -----

        Output: OrderedDict -> {torch.fx.GraphModule.graph.node: [[subgraph_start_index, subgraph_end_index], ...]}
        """
        ir_graph = graph
        graph_nodes = list(ir_graph.nodes)
        subgraph_nodes_0 = [i for i in graph_nodes if len(i.users) > 1 and len(i.args) == 1 and i.op == "call_module"]
        subgraph_nodes_1 = {}
        subgraph_node_map = OrderedDict()

        for sub_node in subgraph_nodes_0:
            dependency_list = []
            dependency_list.append(sub_node)
            for node in graph_nodes:
                if sub_node in node.args:
                    dependency_list.append(node)
            subgraph_nodes_1[sub_node] = dependency_list

        for subnode, subnode_pair in subgraph_nodes_1.items():
            subnode_idx_l = []
            for node in subnode_pair:
                node_idx = graph_nodes.index(node)
                subnode_idx_l.append(node_idx)
            subgraph_node_map[subnode] = [subnode_idx_l[0], subnode_idx_l[-1] + 1]

        def _find_key_after(od, target_key):
            # find the next key in OrderedDict
            keys = list(od.keys())
            index = keys.index(target_key)
            next_key = None
            if index + 1 < len(keys):
                next_key = keys[index + 1]
            else:
                # print('No key found after', target_key)
                pass
            return next_key

        def _insert_key_value(mydict, key, pos_key, value):
            # insert new key-value pair after the target key
            pos = list(mydict.keys()).index(pos_key)
            items = list(mydict.items())
            items.insert(pos + 1, (key, value))
            mydict = OrderedDict(items)
            return mydict

        if subgraph_node_map:
            # turn dependent subgraph to True
            self.subgraph_dependency_bool = True

            # check whether the first key starting index in subgraph_node_map start from 0
            # if not, then add 'header' subgraph to the map
            first_key_map = next(iter(subgraph_node_map))
            if not subgraph_node_map[first_key_map][0] == 0:
                subgraph_node_map.update({'header': [1, subgraph_node_map[first_key_map][0]]})
                subgraph_node_map.move_to_end('header', last=False)

            # check the intermediate node index, if the index between subgraph are not matching with each other
            # then, add intermediate subgraph to the subgraph_node_map
            for key, val in subgraph_node_map.items():
                next_key = _find_key_after(subgraph_node_map, key)
                if next_key:
                    cur_ending_idx = subgraph_node_map[key][-1]
                    nxt_starting_idx = subgraph_node_map[next_key][0]
                    if cur_ending_idx == nxt_starting_idx:
                        continue
                    else:
                        subgraph_node_map[key][-1] = nxt_starting_idx

            # check whehther the last key end index in subgraph_node_map end index matches with the last index number
            # of the graph if not, then add 'end' subgraph to the map
            last_key_map = next(reversed(subgraph_node_map))
            final_node_index = graph_nodes.index(graph_nodes[-1])
            if not subgraph_node_map[last_key_map][1] == final_node_index:
                subgraph_node_map.update({'end': [subgraph_node_map[last_key_map][1],
                                                  final_node_index]})
                subgraph_node_map.move_to_end('end', last=True)

            self.device_range = [1, len(subgraph_node_map)]

        return subgraph_node_map

    def _subgraph_layer_combination(self, graph: torch.fx.GraphModule.graph, split_size: int,
                                    split_option='fixed') -> list:
        """
            This method should be only called when subgraph exists
        """
        subgraph_node_map = self.find_subgraph_user_based(graph)
        result_index = []
        if self.subgraph_dependency_bool:
            # Split model layer index based on option fixed
            if split_option == 'fixed':
                index_pair = [index for key, index in subgraph_node_map.items()]
                index_list = self._split_list(index_pair, split_size, split_option)
                for list_idx in range(len(index_list)):
                    if list_idx > 0:
                        index_list[list_idx].pop(0)
                    result_index.append([index_list[list_idx][0][0], index_list[list_idx][-1][1]])
            elif split_option == "optimized":  # temporarily disabled
                allocation_index = self.model_allocation
                max_split = self.get_max_split(graph)
                index_pair = [index for key, index in subgraph_node_map.items()]
                index_list = self._split_list(index_pair, max_split, "fixed")
                temp_alloc = []
                for list_idx in range(len(index_list)):
                    if list_idx > 0:
                        index_list[list_idx].pop(0)
                    temp_alloc.append([index_list[list_idx][0][0], index_list[list_idx][-1][1]])
                for alloc_index in range(len(allocation_index)):
                    alloc_list = allocation_index[alloc_index]
                    if alloc_list[1] - alloc_list[0] > 1:
                        alloc_start = alloc_list[0]
                        alloc_end = alloc_list[-1] - 1
                    else:
                        alloc_start = alloc_list[0]
                        alloc_end = alloc_list[0]
                    result_index.append([temp_alloc[alloc_start][0], temp_alloc[alloc_end][-1]])

        return result_index

    def get_max_split(self, graph: torch.fx.GraphModule.graph) -> int:
        self.find_subgraph_user_based(graph)
        max_split = self.device_range[-1] - 1
        return max_split

    def subgraph_node_check(self, graph: torch.fx.GraphModule.graph, subgraph_indices: list) -> list:
        """
        Checker function for correcting the operator mismatching issue between Pytorch and ONNX for creating
        Unsqueeze operator.

        :param subgraph_indices: the list of subgraph indices created from the function self.find_subgraph_user_based
        :param graph: model computational graph
        :return: the list of subgraph index that's checked
        """
        # Accessing nodes directly from the graph
        node_list = [i for i in graph.nodes]
        subgraph_nodes = [node_list[idx_list[0]:idx_list[-1]] for idx_list in subgraph_indices]
        subgraph_input_nodes = []
        for subnodes in subgraph_nodes:
            node_names = []
            input_node_names = []
            for node in subnodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.name not in node_names:
                        input_node_names.append(arg.name)
                        node_names.append(arg.name)
                node_names.append(node.name)
            subgraph_input_nodes.append(input_node_names)
        result = []

        for i, current_input_node_names in enumerate(subgraph_input_nodes):
            should_merge = any("reshape" in name for name in current_input_node_names)

            # If the current subgraph should be merged with the previous one
            if should_merge and i > 0:
                previous_indices = result[-1]  # Last item in result
                current_indices = subgraph_indices[i]
                new_indices = [previous_indices[0], current_indices[-1]]
                result[-1] = new_indices  # Update the last item
            else:
                result.append(subgraph_indices[i])

        # Add subgraph index merge to avoid keyerror in subgraph node value remapping. Temp solution.
        final_result = []
        for i, idx_list in enumerate(result):
            start_idx = idx_list[0]
            end_idx = idx_list[-1]
            subnodes = node_list[start_idx:end_idx]
            subgraph = Graph()
            node_names = []
            value_remap = {}
            for node in subnodes:
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.name not in node_names:
                        value_remap[arg] = subgraph.placeholder(arg.name)
                        node_names.append(arg.name)
                node_names.append(node.name)

            for node in subnodes:
                try:
                    value_remap[node] = subgraph.node_copy(node, lambda n: value_remap[n])
                except KeyError: # indicating current model split is not proper, merge with previous subgraph index list
                    if i > 0:
                        # Merge with the previous subgraph
                        previous_start_idx, _ = final_result.pop()
                        idx_list = [previous_start_idx, end_idx]
                        break  # Break the current loop to prevent further processing of this subgraph
            final_result.append(idx_list)

        return final_result

    def get_placeholder_nodes(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n for n in graph.nodes if n.op == "placeholder"]
        return nodes

    def get_placeholder_nodes_name(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n.name for n in graph.nodes if n.op == "placeholder"]
        return nodes

    def get_output_nodes(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n for n in graph.nodes if n.op == "output"]
        return nodes

    def get_output_nodes_name(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n.name for n in graph.nodes if n.op == "output"]
        return nodes

    def get_all_graph_nodes(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n for n in graph.nodes]
        return nodes

    def get_all_graph_nodes_name(self, graph: torch.fx.GraphModule.graph) -> list:
        nodes = [n.name for n in graph.nodes]
        return nodes

    def get_model_graph(self, transformer_model_option):
        graph = None
        if transformer_model_option:
            gm: torch.fx.GraphModule = transformers.utils.fx.symbolic_trace(self.mod)
            graph = gm.graph
        else:
            gm: torch.fx.GraphModule = torch.fx.symbolic_trace(self.mod)
            graph = gm.graph

        assert (graph is not None), "Model Graph cannot be None!"
        return graph
