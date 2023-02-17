import math
from utils.checkpoint import *
from utils.func import *
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from dataset.dataset_urine import UrineSlideDataset
from torch.utils.data import Dataset, DataLoader
import glob
from graph_construction import UrineDataset_for_graph, constructing_graph
from torch_geometric.loader import DataLoader
from GCN_model import GCN
import random

random.seed(1)
torch.manual_seed(1)
def run_vpu(config, loaders, NetworkPhi):
    """
    run VPU.
    :param config: arguments.
    :param loaders: loaders.
    :param NetworkPhi: class of the model.
    """

    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1 # highest test accuracy on test set

    # set up loaders
    (p_loader, x_loader, val_p_loader, val_x_loader, test_loader) = loaders

    # set up vpu model and dataset
    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10','urine']:
        model_phi = NetworkPhi()
        # model_phi = nn.DataParallel(model_phi)
        checkpoint_path = get_checkpoint_path(config)
        checkpoint_path = '/fast/beidi/vpu-tilt/save/128'
        # checkpoint_path = '/fast/beidi/vpu-tilt/save/urineP=[0]_lr=0.0001_lambda=0.03_alpha=0.3'
        checkpoint = torch.load(os.path.join(checkpoint_path, '24.pth'),map_location='cpu')
        model_phi.load_state_dict(checkpoint, strict=False)
        print('Loaded model name:', os.path.join(os.path.join(checkpoint_path, '24.pth')))
        mean_v_samp = torch.Tensor([])
        for p in model_phi.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
        print(mean_v_samp)
        # assert 2==3
        GNN_model = GCN(hidden_channels=64)


    elif config.dataset in ['pageblocks', 'grid', 'avila']:
        input_size = len(p_loader.dataset[0][0])
        model_phi = NetworkPhi(input_size=input_size)
    if torch.cuda.is_available():
        model_phi = model_phi.cuda()

    # set up the optimizer
    lr_phi = config.learning_rate
    opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    GNN_count = 0
    for epoch in range(config.epochs):

        # adjust the optimizer
        # if epoch <= 5 and epoch % 2 == 1:
        #     lr_phi /= 2
        #     print('Learning rate changes to',lr_phi)
        #     # opt_phi = torch.optim.SGD(model_phi.parameters(), lr=lr_phi, momentum=0.9)
        #     opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        # train the model \Phi
        phi_loss, var_loss, reg_loss, phi_p_mean, phi_x_mean = train(config, model_phi, opt_phi, p_loader, x_loader)

        # evaluate the model \Phi
        val_var, test_acc, test_auc = evaluate(config,model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch,
                                              phi_loss, var_loss, reg_loss)

        # assessing performance of the current model and decide whether to save it
        is_val_var_lowest = val_var < lowest_val_var
        is_test_acc_highest = test_acc > highest_test_acc
        lowest_val_var = min(lowest_val_var, val_var)
        highest_test_acc = max(highest_test_acc, test_acc)
        if is_val_var_lowest:
            test_auc_of_best_val = test_auc
            test_acc_of_best_val = test_acc
            epoch_of_best_val = epoch
            best_model = model_phi.state_dict()
        torch.save(best_model, checkpoint_path + '/' + str(epoch+25) + '.pth')

        # train_acc, train_loss = gcn_train(model, GNN_train_loader, criterion, optimizer, scheduler)
        # test_acc, test_loss = gcn_test(model, GNN_test_loader, criterion)
        # print(f'GNN - Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        if epoch == 0:
            test_best_acc = test_acc
            GNN_model_save = GNN_model
        if test_acc <= test_best_acc:
            GNN_count += 1
        # if (epoch+1)
        # N_best_acc = Graph_classification(model_phi,GNN_model,config,epoch)
        # slide_predict(model_phi, config, x_loader)
    # inform users model in which epoch is finally picked
    GNN_model_save,GNN_best_acc = Graph_classification(model_phi, GNN_model, config, 44)
    print('Early stopping at {:}th epoch, test AUC : {:.4f}, test acc: {:.4f}'.format(epoch_of_best_val, test_auc_of_best_val, test_acc_of_best_val))
    # print('Load model of epoch',epoch_of_best_val )
    # model_phi.load_state_dict(best_model)



def train(config, model_phi, opt_phi, p_loader, x_loader):
    """
    One epoch of the training of VPU.

    :param config: arguments.
    :param model_phi: current model \Phi.
    :param opt_phi: optimizer of \Phi.
    :param p_loader: loader for the labeled positive training data.
    :param x_loader: loader for training data (including positive and unlabeled)
    """
    # setup some utilities for analyzing performance
    phi_p_avg = AverageMeter()
    phi_x_avg = AverageMeter()
    phi_loss_avg = AverageMeter()
    var_loss_avg = AverageMeter()
    reg_avg = AverageMeter()

    # set the model to train mode
    model_phi.train()

    for batch_idx in range(config.val_iterations):
        try:
            data_x, _ = next(x_iter)
        except:
            x_iter = iter(x_loader)
            data_x, _ = next(x_iter)

        try:
            data_p, _ = next(p_iter)
        except:
            p_iter = iter(p_loader)
            data_p, _ = next(p_iter)

        if torch.cuda.is_available():
            data_p, data_x = data_p.cuda(), data_x.cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_x))
        output_phi_all,_ = model_phi(data_all)
        log_phi_all = output_phi_all[:, 1]
        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        output_phi_x = output_phi_all[idx_x]
        var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)

        # perform Mixup and calculate the regularization
        target_x = output_phi_x[:, 1].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda() if torch.cuda.is_available() else target_p
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(config.mix_alpha, config.mix_alpha)
        lam = m.sample()
        data = lam * data_x + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        # print('target.size()',target.size())
        out_log_phi_all, _ = model_phi(data)
        # print('out_log_phi_all.size()',out_log_phi_all[:,1].size())
        reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

        # calculate gradients and update the network
        phi_loss = var_loss + config.lam * reg_mix_log
        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()
        # scheduler.step()
        # update the utilities for analysis of the model
        reg_avg.update(reg_mix_log.item())
        phi_loss_avg.update(phi_loss.item())
        var_loss_avg.update(var_loss.item())
        phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        phi_x_avg.update(phi_x.mean().item(), len(phi_x))
    return phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg

def evaluate(config, model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch, phi_loss, var_loss, reg_loss):
    """
    evaluate the performance on test set, and calculate the variational loss on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param test_loader: loader for the test set (fully labeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to var_loss + reg_loss.
    :param var_loss: variational loss of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()

    # calculate variational loss of the validation set consisting of PU data
    val_var = cal_val_var(model_phi, val_p_loader, val_x_loader)

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (data, _) in enumerate(x_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        new_log_max_phi,_ = model_phi(data)
        log_max_phi = max(log_max_phi, new_log_max_phi[:, 1].max())

    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            log_phi,_ = model_phi(data)
            log_phi = log_phi[:, 1]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))
    # pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
    pred_all = np.array((log_phi_all > math.log(config.th)).cpu().detach())
    # print(pred_all)
    log_phi_all = np.array(log_phi_all.cpu().detach())
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all)
    test_auc = roc_auc_score(target_all, log_phi_all)
    print('Train Epoch: {}\t phi_loss: {:.4f}   var_loss: {:.4f}   reg_loss: {:.4f}   Test accuracy: {:.4f}   Val var loss: {:.4f}' \
          .format(epoch, phi_loss, var_loss, reg_loss, test_acc, val_var))
    return val_var, test_acc, test_auc

def slide_predict(model_phi,args,x_loader):
    model_phi.eval()
    import glob
    rootDir ='/bigdata/projects/beidi/dataset/urine/Urine_divide/test'
    name = []
    name = glob.glob(rootDir + '/*' + '/*')
    index_list = list(range(len(name)))
    label_list = []
    pred_list = []
    num_correct = 0
    count_cancer, count_benign, count_atypical, count_suspicious = 0, 0, 0, 0
    num_correct_cancer, num_correct_benign, num_correct_atypical, num_correct_suspicious = 0, 0, 0, 0
    positive_label_list = [0]
    target_transform = lambda x: 1 if x in positive_label_list else 0
    for i in name:
        print('Slide',i.split('/')[-1],'is', i.split('/')[-2])
        file_path = os.path.join(rootDir,i)
        test_dataset = UrineSlideDataset(dataset_path=file_path,target_transform=target_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        if i.split('/')[-2] == 'benign':
            label = 1
            label_list.append(1)
            count_benign += 1
        elif i.split('/')[-2] == 'atypical':
            label = 1
            label_list.append(1)
            count_atypical += 1
        elif i.split('/')[-2] == 'cancer':
            label = 0
            label_list.append(0)
            count_cancer += 1
        elif i.split('/')[-2] == 'suspicious':
            label = 0
            label_list.append(0)
            count_suspicious += 1
        else:
            assert 2 == 3
        # get cell embedding

        log_max_phi = -math.inf
        for idx, (data, _) in enumerate(x_loader):
            if torch.cuda.is_available():
                data = data.cuda()
            new_log_max_phi,_ = model_phi(data)
            log_max_phi = max(log_max_phi, new_log_max_phi[:, 1].max())
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                pred, _ =  model_phi(data)
                log_phi = pred[:, 1]
                log_phi -= log_max_phi
                if idx == 0:
                    log_phi_all = log_phi
                    target_all = target
                else:
                    log_phi_all = torch.cat((log_phi_all, log_phi))
                    target_all = torch.cat((target_all, target))
        # print(log_phi_all.size())
        # print(log_phi_all)
        pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
        pred_all_inverse = np.ones(pred_all.shape) - pred_all
        # print(pred_all_inverse)
        count_cancer_cell = pred_all_inverse.sum()
        pred_list.append(count_cancer_cell.item())
        print('count cancer cell {:.0f}'.format(count_cancer_cell))
        if count_cancer_cell > len(pred_all_inverse)/2: ## adjust threshold
            slide_pred = 0
        else:
            slide_pred = 1

        if slide_pred == label:
            num_correct += 1
            if i.split('/')[-2] == 'benign':
                num_correct_benign += 1
            elif i.split('/')[-2] == 'atypical':
                num_correct_atypical += 1
            elif i.split('/')[-2] == 'cancer':
                num_correct_cancer += 1
            elif i.split('/')[-2] == 'suspicious':
                num_correct_suspicious += 1
            else:
                assert 2 == 3
        # else:
        #     print('False')
    num_correct = num_correct_suspicious + num_correct_atypical + num_correct_cancer + num_correct_suspicious
    slide_acc = num_correct/len(name)
    print('Slide level acc is {:.4f}'.format(slide_acc))
    print('Acc of benign is {:.4f}'.format(num_correct_benign/count_benign))
    print('Acc of atypical is {:.4f}'.format(num_correct_atypical/count_atypical))
    print('Acc of suspicious is {:.4f}'.format(num_correct_suspicious/count_suspicious))
    print('Acc of cancer is {:.4f}'.format(num_correct_cancer/count_cancer))
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    ROC_curve(label_list,pred_list,save_name='/slide_roc')


def cal_val_var(model_phi, val_p_loader, val_x_loader):
    """
    Calculate variational loss on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, (data_x, _) in enumerate(val_x_loader):
            if torch.cuda.is_available():
                data_x = data_x.cuda()
            output_phi_x_curr,_ = model_phi(data_x)
            if idx == 0:
                output_phi_x = output_phi_x_curr
            else:
                output_phi_x = torch.cat((output_phi_x, output_phi_x_curr))
        for idx, (data_p, _) in enumerate(val_p_loader):
            if torch.cuda.is_available():
                data_p = data_p.cuda()
            output_phi_p_curr,_ = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_x = output_phi_x[:, 1]
        var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - torch.mean(log_phi_p)
        return var_loss.item()

def ROC_curve(test_labels, test_score,save_name):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    fpr, tpr, thresholds = roc_curve(test_labels, test_score)
    print('AUC: {}'.format(auc(fpr, tpr)))
    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i / 20.0 for i in range(21)])
    plt.xticks([i / 20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('/bigdata/projects/beidi/git/vpu-slide/result' + save_name )
    # np.save('/bigdata/projects/beidi/git/vpu-slide/result'+ save_name+ "_test_score.npy", test_score)


def gcn_train(model,train_loader,criterion,optimizer,scheduler):
    correct = 0
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(train_loader.dataset), loss

def gcn_test(model,loader,criterion):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    correct = 0
    model.eval()
    for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)  # Compute the loss.
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         # print('pred',pred)
         # print('y   ',data.y)
         # Confusion Matrix
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix:')
    confu_matrix = confusion_matrix(data.y.tolist(), pred.tolist())
    print(confu_matrix)
    print('AUC:', roc_auc_score(data.y.tolist(), pred.tolist()))
    print('Recall:', recall_score(data.y.tolist(), pred.tolist()))
    print('Precision:', precision_score(data.y.tolist(), pred.tolist()))
    print('F1 score:', f1_score(data.y.tolist(), pred.tolist()))
    print('Sensitivity:', confu_matrix[0,0]/(confu_matrix[0,0]+confu_matrix[1,0]))
    print('Specificity:', confu_matrix[1,1]/(confu_matrix[0,1]+confu_matrix[1,1]))
    # ROC
    # ROC_curve(data.y.tolist(), pred.tolist(), save_name='/slide_roc_GNN')
    return correct / len(loader.dataset), loss, model  # Derive ratio of correct predictions.

def Graph_classification(model_phi,model,config,epoch):

    # slide_predict(model, device)
    # slide_root = '/bigdata/projects/beidi/dataset/urine/Urine_divide'
    slide_root = '/fast/beidi/data/tile128_rand100'
    graph_root = '/fast/beidi/crossvit/data/scale128'
    slide_train = os.path.join(slide_root, 'train')
    slide_test = os.path.join(slide_root, 'test')
    graph_train = os.path.join(graph_root, 'train')
    graph_test = os.path.join(graph_root, 'test')
    train_name = glob.glob(slide_train + '/*' + '/*')
    test_name = glob.glob(slide_test + '/*' + '/*')
    # print(train_name)

    shutil.rmtree(graph_root)
    os.makedirs(graph_train)
    os.makedirs(graph_test)
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    plt.figure(figsize=(8, 8))
    for i in train_name:
        print(i)
        file_path = os.path.join(slide_train, i)
        train_dataset = UrineSlideDataset(dataset_path=file_path)
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=4)
        embeddings_train,pred = constructing_graph(train_loader, model_phi)
        normal_idxs = (pred == 0)
        abnorm_idxs = (pred == 1)
        tsne_normal = embeddings_train[normal_idxs]
        tsne_abnormal = embeddings_train[abnorm_idxs]
        if i.split('/')[-2]=='cancer':
            marker = 'x'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in cancer')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in cancer')

        elif i.split('/')[-2]=='benign':
            marker = 's'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in benign')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in benign')

        elif i.split('/')[-2]=='suspicious':
            marker = '^'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in suspicious')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in suspicious')

        elif i.split('/')[-2]=='atypical':
            marker = 'o'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in atypical')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in atypical')

        print('Predicting slide {}, the embedding shape is {}'.format(i.split('/')[-1],embeddings_train.shape))
        if not os.path.exists(os.path.join(graph_train,'cancer')):
            os.makedirs(os.path.join(graph_train,'cancer'))
        if not os.path.exists(os.path.join(graph_train,'benign')):
            os.makedirs(os.path.join(graph_train, 'benign'))
        if not os.path.exists(os.path.join(graph_train, 'atypical')):
            os.makedirs(os.path.join(graph_train, 'atypical'))
        if not os.path.exists(os.path.join(graph_train, 'suspicious')):
            os.makedirs(os.path.join(graph_train, 'suspicious'))
        np.save(graph_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_train)
        # np.save(graph_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)

    for i in test_name:
        print(i)
        file_path = os.path.join(slide_test, i)
        test_dataset = UrineSlideDataset(dataset_path=file_path)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)
        embeddings_test,pred = constructing_graph(test_loader, model_phi)
        normal_idxs = (pred == 0)
        abnorm_idxs = (pred == 1)
        tsne_normal = embeddings_test[normal_idxs]
        tsne_abnormal = embeddings_test[abnorm_idxs]
        if i.split('/')[-2] == 'cancer':
            marker = 'x'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in cancer')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in cancer')

        elif i.split('/')[-2] == 'benign':
            marker = 's'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in benign')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in benign')

        elif i.split('/')[-2] == 'suspicious':
            marker = '^'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in suspicious')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in suspicious')

        elif i.split('/')[-2] == 'atypical':
            marker = 'o'
            plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], s=20, linewidths=1, alpha=.6, color='red', marker=marker,
                        label='cancer in atypical')
            plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], s=20, linewidths=1, alpha=.6, color='green',
                        marker=marker, label='benign in atypical')
        print('Predicting slide {}, the embedding shape is {}'.format(i.split('/')[-1],embeddings_test.shape))
        if not os.path.exists(os.path.join(graph_test,'cancer')):
            os.makedirs(os.path.join(graph_test,'cancer'))
        if not os.path.exists(os.path.join(graph_test,'benign')):
            os.makedirs(os.path.join(graph_test, 'benign'))
        if not os.path.exists(os.path.join(graph_test, 'atypical')):
            os.makedirs(os.path.join(graph_test, 'atypical'))
        if not os.path.exists(os.path.join(graph_test, 'suspicious')):
            os.makedirs(os.path.join(graph_test, 'suspicious'))
        np.save(graph_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_test)
        # np.save(graph_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=10)
    # plt.tick_params(labelsize=20)
    # int_path = get_checkpoint_path(config)
    # plt.savefig(os.path.join(checkpoint_path,str(epoch+1)+'_tsne'))


    train_dataset = UrineDataset_for_graph(root=graph_train)
    test_dataset = UrineDataset_for_graph(root=graph_test)
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    test_best = 0
    count = 0
    for epoch in range(60):
        train_acc, train_loss = gcn_train(model, train_loader, criterion, optimizer,scheduler)
        # train_acc, train_loss = gcn_test(model, train_loader, criterion)
        test_acc, test_loss,GNN_model = gcn_test(model, test_loader, criterion)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        if epoch == 0:
            test_best_acc = test_acc
        if test_acc <= test_best_acc:
            test_best_acc = test_acc
    return GNN_model, test_best_acc