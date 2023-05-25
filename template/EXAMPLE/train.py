# -*- coding: utf-8 -*-
"""
@Time: 2023/5/9 19:28 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

"""
import the module you need

def train(args, data, logger):
    # args includes many parameters, if necessary, you can specify them here. such as:
    args.embedding_dim = 10
    arg.hidden_dim = 256
    # Even you can specify the hyper-parameters like lr, epoch, lambda, according to your model.
    # For convenience, you can use a dict to store them.
    params_dict = { "acm": [50, 1e-3],
                    "cite": [50, 1e-4],
                    ... }
    # Then
    args.max_epoch = params_dict[args.dataset_name][0]
    args.lr = params_dict[args.dataset_name][1]
    
    # load model
    model = Example(arg1, arg2, ...)
    logger.info(model)
    
    # If the model was pretrained, such as only ae module was pretrained:
    # pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    # model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    
    # If there are many modules were pretrained, such as ae and gae were pretrained:
    # pretrain_ae_filename = args.pretrain_ae_save_path + args.dataset_name + ".pkl"
    # pretrain_gae_filename = args.pretrain_gae_save_path + args.dataset_name + ".pkl"
    # ...
    # model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    # model.gae.load_state_dict(torch.load(pretrain_gae_filename, map_location='cpu'))
    # ...
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    feature = data.feature 
    label = data.label
    adj = data.adj  # Please note the adj is symmetric normalized or asymmetric normalized.
    
    with torch.no_grad():
        _, _, _, _, z = model.ae(feature)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    
    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch+1):
        model.train()
        # train
        x_bar, q, pred, _, embedding = model(feature, adj)
        
        # loss
        re_loss = F.mse_loss(x_bar, feature)
        loss = lambda_1 * kl_loss + lambda_2 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculate metrics
        with torch.no_grad():
            model.eval()
            _, _, pred, _, _ = model(feature, adj)
            y_pred = pred.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, y_pred)
            # record the max value
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    
    result = Result(embedding=embedding, acc_max_corresponding_metrics=acc_max_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    
    # the format of return is solid, if you want to change, remember to change the corresponding place in main.py
    return result
"""
