import copy
import torch
from sklearn.cluster import AgglomerativeClustering

def WeightedAvgWeights(weights_list, client_weights, layer_names_to_avg, selected_clients, common_params):
    """
    Perform weighted averaging on the specified layers for the selected clients.

    :param weights_list: List of model weights for each client.
    :param client_weights: List of weights for each client.
    :param layer_names_to_avg: List of layer names to aggregate (e.g., 'sam_layers' or 'mem_layers').
    :param selected_clients: List of indices representing the clients to include in averaging.
    :param common_params: Set of common parameters to aggregate (for non-task-specific case).
    :return: Aggregated weights for the selected layers.
    """
    # Initialize a dictionary for the aggregated parameters
    aggregated_weights = {}

    for key in common_params:
        # Only aggregate parameters in the selected layers
        if 'site_gating'  in key:
            print("Skip: ", key)
            continue
        if 'site_embedding_layers' in key:
            print("Skip: ", key)
            continue
        if any(layer_name in key for layer_name in layer_names_to_avg):
            # Initialize with zero for aggregation
            aggregated_param = torch.zeros_like(weights_list[0][key])

            # Perform weighted sum of the layer parameters across the selected clients
            total_weight = 0.0
            for i in selected_clients:
                aggregated_param += weights_list[i][key] * client_weights[i]
                total_weight += client_weights[i]

            # Normalize by total weight if needed
            if total_weight > 0:
                aggregated_param /= total_weight

            # Store the aggregated parameter
            aggregated_weights[key] = aggregated_param

    return aggregated_weights

def FedAvg(weights_list, num_clients, Tasks, Layers='both', client_weights=None, Task_specific=False):
    """
    Federated Averaging for sam_layers, mem_layers, or both, returning a new weights list.

    :param weights_list: List of model weights for each client.
    :param num_clients: Number of clients participating in the aggregation.
    :param Tasks: List of tasks corresponding to each client, e.g., 'dep' or 'seg'.
    :param Layers: Specify 'sam', 'mem', or 'both' to determine which layers to aggregate.
        'sam' == FedAvg
        'mem' == FedRep
    :param client_weights: List of weights for each client (for weighted averaging). If None, equal weights are used.
    :param Task_specific: If True, only average models with the same task.
    :return: A new weights_list where sam_layers and/or mem_layers are averaged across clients.
    """
    # If no client_weights are provided, assign equal weights to each client
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients

    # Define the parameter names related to sam_layers and mem_layers
    sam_layer_names = ['sam_mask_decoder']
    mem_layer_names = ['obj_ptr_proj', 'memory_encoder', 'memory_attention', 'mask_downsample']

    # Select the layer names based on the Layers argument
    if Layers == 'sam':
        layer_names_to_avg = sam_layer_names
    elif Layers == 'mem':
        layer_names_to_avg = mem_layer_names
    else:  # 'both'
        layer_names_to_avg = sam_layer_names + mem_layer_names

    # Task-specific case: group clients by their task type and perform averaging separately
    if Task_specific:
        task_aggregated_weights = {}  # Dictionary to store aggregated weights for each task type
        unique_tasks = set(Tasks)

        for task in unique_tasks:
            # Select clients with the current task
            selected_clients = [i for i in range(num_clients) if Tasks[i] == task]
            
            # Perform weighted average for the selected clients
            task_common_params = set(weights_list[selected_clients[0]].keys())
            for client_weights_dict in [weights_list[i] for i in selected_clients]:
                task_common_params.intersection_update(client_weights_dict.keys())

            # Perform the weighted averaging for the specific task
            task_aggregated_weights[task] = WeightedAvgWeights(weights_list, client_weights, layer_names_to_avg, selected_clients, task_common_params)

        # Apply the task-specific aggregated weights back to the corresponding clients
        for i in range(num_clients):
            task = Tasks[i]
            for key, value in task_aggregated_weights[task].items():
                weights_list[i][key] = value

    else:
        # Precompute common parameters across all clients when Task_specific is False
        common_params = set(weights_list[0].keys())
        for client_weights_dict in weights_list[1:]:
            common_params.intersection_update(client_weights_dict.keys())

        # Remove task-specific parameters (e.g., depth-related parameters for 'seg' tasks)
        for i in range(num_clients):
            if Tasks[i] == 'dep':
                task_specific_params = {k for k in weights_list[i].keys() if 'depth' in k}
                common_params.difference_update(task_specific_params)

        # Perform global weighted averaging
        selected_clients = list(range(num_clients))
        aggregated_weights = WeightedAvgWeights(weights_list, client_weights, layer_names_to_avg, selected_clients, common_params)

        # Apply the global aggregated weights to all clients
        for i in range(num_clients):
            for key, value in aggregated_weights.items():
                weights_list[i][key] = value

    return weights_list

def MaTFL(weights_list, num_clients, Layers='both', client_weights=None, cluster_num=2):
    """
    MaTFL: A new federated learning aggregation strategy based on model clustering.

    :param weights_list: List of model weights for each client.
    :param num_clients: Number of clients participating in the aggregation.
    :param Layers: Specify 'sam', 'mem', or 'both' to determine which layers to aggregate.
    :param client_weights: List of weights for each client (for weighted averaging). If None, equal weights are used.
    :param cluster_num: Number of clusters for the grouping strategy.
    :return: A new weights_list where layers are aggregated according to MaTFL strategy.
    """
    # If no client_weights are provided, assign equal weights to each client
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients

    # Define the parameter names related to sam_layers and mem_layers
    sam_layer_names = ['sam_mask_decoder']
    mem_layer_names = ['obj_ptr_proj', 'memory_encoder', 'memory_attention', 'mask_downsample']

    # Select the layer names based on the Layers argument
    if Layers == 'sam':
        layer_names_to_avg = sam_layer_names
    elif Layers == 'mem':
        layer_names_to_avg = mem_layer_names
    else:  # 'both'
        layer_names_to_avg = sam_layer_names + mem_layer_names

    # Get keys shared across all clients
    common_params = set(weights_list[0].keys())
    for client_weights_dict in weights_list[1:]:
        common_params.intersection_update(client_weights_dict.keys())

    # Perform grouping based on model clustering using deltas
    scores = get_grouping_score(weights_list, common_params, cluster_num)

    # Perform the weighted averaging based on the grouping scores
    updated_weights_list = copy.deepcopy(weights_list)
    for i in range(num_clients):
        for key in common_params:
            if any(layer_name in key for layer_name in layer_names_to_avg):
                aggregated_param = torch.zeros_like(weights_list[i][key])
                total_weight = 0.0
                for j in range(num_clients):
                    aggregated_param += weights_list[j][key] * scores[i, j]
                    total_weight += scores[i, j]
                if total_weight > 0:
                    aggregated_param /= total_weight
                updated_weights_list[i][key] = aggregated_param

    return updated_weights_list

def get_grouping_score(ckpts, keys, cluster_num):
    """
    Perform clustering based on the difference between each model and the average model (model soup).
    
    :param ckpts: Dictionary of model checkpoints (each model's state_dict).
    :param keys: List of keys to consider (shared keys between models).
    :param cluster_num: Number of clusters for the grouping.
    :return: A similarity score matrix based on clustering results.
    """
    # Step 1: Calculate the model soup (average model)
    model_soup = get_model_soup(ckpts, keys)

    delta_list = []
    for key in keys:
        # Compute the difference between each model's parameter and the model soup
        temp_delta = torch.stack([ckpt[key] for ckpt in ckpts], dim=0) - model_soup[key]
        delta_list.append(temp_delta.reshape([len(temp_delta), -1]))

    # Step 2: Concatenate all the deltas into a single tensor
    delta = torch.cat(delta_list, dim=1)
    # Turn the delta tensor into a float tensor
    delta = delta.float()
    # Step 3: Perform Agglomerative Clustering on the delta using cosine similarity
    clustering = AgglomerativeClustering(n_clusters=cluster_num, metric='cosine', linkage='average').fit(delta.cpu())
    print(clustering.labels_)

    # Step 4: Convert clustering labels to tensor
    cluster_results = torch.tensor(clustering.labels_).cuda()

    # Step 5: Calculate the similarity score based on clustering labels
    scores = torch.eq(cluster_results.view(-1, 1), cluster_results.view(1, -1)).float()
    scores = scores / scores.sum(dim=1, keepdim=True)
    
    return scores

def get_model_soup(ckpts, keys):
    """
    Calculate the 'model soup', which is the average of all models' weights for the given keys.
    
    :param ckpts: Dictionary of model checkpoints (each model's state_dict).
    :param keys: List of keys to consider (shared keys between models).
    :return: A dictionary representing the average model weights for the given keys.
    """
    model_soup = {}
    for key in keys:
        # Stack the weights for the given key and compute the mean across all models
        key_weights = torch.stack([ckpt[key] for ckpt in ckpts], dim=0)
        model_soup[key] = torch.mean(key_weights, dim=0)
    
    return model_soup

def Hyper_Aggregate(now_wieghts_list, last_weight_list, num_clients, hyper_optimizer, hypernetwork, last_ckpts):
    delta_dict_list = []  
    
    for i in range(num_clients):
        delta_dict = {}
        for key in now_wieghts_list[i].keys():
            delta_dict[key] = now_wieghts_list[i][key] - last_ckpts[i][key]
        delta_dict_list.append(delta_dict)
    
    aggregated_weights = hypernetwork(last_ckpts, delta_dict_list)

    del delta_dict_list
    # The output of this func is also the last_weight_list for the next round
    return aggregated_weights

def HyperCrossAttention_Update(now_wieghts_list, last_weight_list, num_clients, hyper_optimizer, hypernetwork, last_ckpts):
    """
    Aggregates weights across clients using HyperCrossAttention and updates the hypernetwork.
    
    Args:
        now_wieghts_list (list): A list of parameter dictionaries from clients after HyperCrossAttention.
        last_weight_list (list): A list of parameter dictionaries from clients before HyperCrossAttention.
        num_clients (int): Number of clients involved in aggregation.
        hyper_optimizer (torch.optim.Optimizer): The optimizer for the hypernetwork.
        hypernetwork (nn.Module): The HyperCrossAttention model to be updated.
    
    Returns:
        Updated weights after aggregation.
    """

    delta_dict_list = []  

    for i in range(num_clients):
        delta_dict = {}
        for key in now_wieghts_list[i].keys():
            delta_dict[key] = now_wieghts_list[i][key] - last_ckpts[i][key]
        delta_dict_list.append(delta_dict)

    hypernetwork.train()
    hyper_optimizer.zero_grad()

    for i in range(num_clients):
        last_output_list = []
        diff_param = []

        for key in last_weight_list[i].keys():
            if key in hypernetwork.common_keys:
                last_output_list.append(last_weight_list[i][key])
                diff_param.append(delta_dict_list[i][key])
        
        
        torch.autograd.backward(last_output_list, diff_param, retain_graph=True)

    hyper_optimizer.step()

    del delta_dict_list
    if last_output_list:
        print("Training on the server side Hypernetwork")

