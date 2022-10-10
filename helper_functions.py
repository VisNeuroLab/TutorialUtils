import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def conv_visualize(weights, num_inh_list=[0]):
    # let num_inh_list be num_inh for all prev layers, with ind -1 for visualized layer
    
    plt.figure()
    plt.subplot()
    # plt.title("LBFGS - Layer 0")
    sx = int(np.ceil(np.sqrt(weights.shape[-1])))
    sy = int(np.round(np.sqrt(weights.shape[-1])))

    outer = gridspec.GridSpec(sx, sy, wspace=0.2, hspace=0.2)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)
    
    for cc in range(weights.shape[-1]):        
        vmin, vmax = weights[:,:,cc].min(), weights[:,:,cc].max()
        if len(num_inh_list) <= 1:
            plt.subplot(outer[cc])
            plt.imshow(weights[:,:,cc], interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)
            if cc < weights.shape[-1] - sy:
                plt.gca().set_xticks([])
            if cc % sy != 0:
                plt.gca().set_yticks([])
            if cc >= weights.shape[-1] - num_inh_list[-1]:
                for i in plt.gca().spines.items():
                    i[-1].set_color('red')
        else:
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[cc], wspace=0.1, hspace=0)
            plt.subplot(inner[0])
            plt.imshow(weights[:(len(weights)-num_inh_list[-2]),:,cc], interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)
            plt.gca().set_xticks([])
            if cc % sy != 0:
                plt.gca().set_yticks([])
            if cc >= weights.shape[-1] - num_inh_list[-1]:
                for i in plt.gca().spines.items():
                    i[-1].set_color('red')    
            plt.subplot(inner[1])
            plt.imshow(weights[num_inh_list[-2]:,:,cc], interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)    
            if cc < weights.shape[-1] - sy:
                plt.gca().set_xticks([])
            if cc % sy != 0:
                plt.gca().set_yticks([])
            for i in plt.gca().spines.items():
                i[-1].set_color('red')      
        
    print(weights.shape)

def eval_model(model, valid_dl):
    loss = model.loss.unit_loss
    model.eval()

    LLsum, Tsum, Rsum = 0, 0, 0
    from tqdm import tqdm
        
    device = next(model.parameters()).device  # device the model is on
    if isinstance(valid_dl, dict):
        for dsub in valid_dl.keys():
                if valid_dl[dsub].device != device:
                    valid_dl[dsub] = valid_dl[dsub].to(device)
        rpred = model(valid_dl)
        LLsum = loss(rpred,
                    valid_dl['robs'][:,model.cids],
                    data_filters=valid_dl['dfs'][:,model.cids],
                    temporal_normalize=False)
        Tsum = valid_dl['dfs'][:,model.cids].sum(dim=0)
        Rsum = (valid_dl['dfs'][:,model.cids]*valid_dl['robs'][:,model.cids]).sum(dim=0)

    else:
        for data in tqdm(valid_dl, desc='Eval models'):
                    
            for dsub in data.keys():
                if data[dsub].device != device:
                    data[dsub] = data[dsub].to(device)
            
            with torch.no_grad():
                rpred = model(data)
                LLsum += loss(rpred,
                        data['robs'][:,model.cids],
                        data_filters=data['dfs'][:,model.cids],
                        temporal_normalize=False)
                Tsum += data['dfs'][:,model.cids].sum(dim=0)
                Rsum += (data['dfs'][:,model.cids] * data['robs'][:,model.cids]).sum(dim=0)
                
    LLneuron = LLsum/Rsum.clamp(1)

    rbar = Rsum/Tsum.clamp(1)
    LLnulls = torch.log(rbar)-1
    LLneuron = -LLneuron - LLnulls

    LLneuron/=np.log(2)

    return LLneuron.detach().cpu().numpy()

def visualize_native_conv1d_layer(weights):
  sx = int(np.ceil(np.sqrt(weights.shape[0])))
  sy = int(np.round(np.sqrt(weights.shape[0])))
  plt.figure(figsize=(sx, sy))
  
  for cc in range(weights.shape[0]):
      plt.subplot(sx, sy, cc+1)
      plt.imshow(weights[cc, :, :], interpolation='none', aspect='auto')

  plt.tight_layout()
  
def lin_visualize(weights, split=None, num_eni_list=[]):
    plt.figure()
    plt.subplot()
    # plt.title("LBFGS - Layer 0")
    sx = int(np.ceil(np.sqrt(weights.shape[-1])))
    sy = int(np.round(np.sqrt(weights.shape[-1])))

    outer = gridspec.GridSpec(sx, sy, wspace=0.2, hspace=0.2)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)
    
    for cc in range(weights.shape[-1]):
        vmin, vmax = weights[:,:,cc].min(), weights[:,:,cc].max()
        ratios = [sum(i)/len(weights) for i in num_eni_list]
        inner = gridspec.GridSpecFromSubplotSpec(len(num_eni_list), 1, subplot_spec=outer[cc], wspace=0.1, hspace=0.05, height_ratios=ratios)
        ind = 0
        for i in range(len(num_eni_list)):
            e_num, i_num = num_eni_list[i]
            num_splits = int(e_num > 0) + int(i_num > 0)
            inner_ratios = [1] if e_num == 0 or i_num == 0 else [e_num/(e_num+i_num), i_num/(e_num+i_num)]
            inner_split = gridspec.GridSpecFromSubplotSpec(num_splits, 1, subplot_spec=inner[i], wspace=0.1, hspace=0, height_ratios=inner_ratios)
            if e_num > 0:
                plt.subplot(inner_split[0])
                plt.imshow(weights[ind:ind+e_num,:,cc], interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)
                ind += e_num
                if num_splits == 2 or cc < weights.shape[-1] - sy:
                    plt.gca().set_xticks([])
                if cc % sy != 0:
                    plt.gca().set_yticks([])
            if i_num > 0:
                plt.subplot(inner_split[-1])
                plt.imshow(weights[ind:ind+i_num,:,cc], interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)
                ind += i_num
                if cc < weights.shape[-1] - sy:
                    plt.gca().set_xticks([])
                if cc % sy != 0:
                    plt.gca().set_yticks([])      
                for j in plt.gca().spines.items():
                    j[-1].set_color('red')    