import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_prune_example(images_loader: torch.utils.data.DataLoader,
                       data_scores:torch.Tensor,
                       hardest:bool=True,
                       range_:float=.2,
                       random:bool=True,
                       prune_method_name:str=''):
    """
    plot a bunch of examples

    Args:
        images_loader: data loader for images
        data_scores: scores for each example how much is hard to learn
        hardest: True/False -> plot the hardest/easiest
        range_ (float in range (0,1]): range to take random example from for example:
            range=.5 -> take random example from hardest/easiest examples
        random: plot random example from chosen range, if false plot the hardest/simplest examples
    """
    plot_num = 12
    choices=np.random.choice(int(range_*data_scores.shape[0]),plot_num,replace=False) if random else np.arange(plot_num)
    idx=data_scores.sort(descending=hardest)[1][choices]

    plt.style.use('default')
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    fig.suptitle(f"{'Hardest' if hardest else 'Easiest'} examples")
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    for ax, i in zip(axes.reshape(-1), idx):
        ax.imshow(images_loader[i][0])
        ax.set_title(f"{prune_method_name+' '}{data_scores[i]:.3f}, "
                     f"Class: {images_loader.classes[images_loader[i][1]]}")
    plt.show()
