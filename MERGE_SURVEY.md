# Model Merging: A Survey
via: https://cameronrwolfe.substack.com/p/model-merging
### From modern LLM applications to the early days of machine learning research...

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Sep 16, 2024

63

[

8

](https://cameronrwolfe.substack.com/p/model-merging/comments)

8

Share

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F452256f9-a822-46b7-96b4-3bec0913af2e_2556x1430.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F452256f9-a822-46b7-96b4-3bec0913af2e_2556x1430.png)

(from \[1, 2, 8, 9, 10, 12, 15\])

To improve the performance of a machine learning model, we can train several models independently and average their predictions at inference time to form an ensemble. Ensembling has been used for decades in machine learning, but this approach comes with the downside of increased inference costs—_we must compute the output of several models!_[1](https://cameronrwolfe.substack.com/p/model-merging#footnote-1-147448898) To avoid this issue, researchers have explored alternative techniques for combining models. These explorations eventually led to the popularization of weight-space ensembles, which average the weights of the models in an ensemble—_forming a single, merged model_—instead of averaging their predictions. This technique was found to perform quite well, matching or exceeding the performance of vanilla, output-space ensembles in many cases.

> _“We unveil that Language Models (LMs) can acquire new capabilities by assimilating parameters from homologous models without retraining or GPUs.”_ - from \[3\]

Today, model merging is a popular research topic, but this idea is not new whatsoever—_we can trace it all the way back to the 1990s_ \[7\]! In the deep learning era[2](https://cameronrwolfe.substack.com/p/model-merging#footnote-2-147448898), techniques related to model merging have repeatedly appeared in research topics like mode connectivity, generalization, continual learning and more. In the last few years especially, the level of interest in model merging has exploded due to its effectiveness in applications with large language models (LLMs). We have seen model merging used to combine the capabilities of several foundation models, inject new skills into a model, or even improve the alignment process. In this overview, we will take a deep look at all of this research, starting from the beginning and working our way up to modern applications with LLMs.

Foundations and Background Information
--------------------------------------

Before diving into recent research on model merging, we will take a look at the early work in this space. Additionally, we will explore a few different, but related, research topics that from the basis of model merging. By better understanding these techniques and their origins, we will gain a more nuanced perspective on model merging techniques, allowing us to more deeply understand the core ideas in this space, where they come from, and why they work so well.

#### The Origins of Model Merging

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd600b414-3b2a-480b-a1e9-625af92f7d62_2016x1168.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd600b414-3b2a-480b-a1e9-625af92f7d62_2016x1168.png)

(from \[7\])

Model merging is a popular research topic as of late, but the history of this technique is quite extensive, dating all the way back to the mid 1990s! Authors in \[7\] observe that practitioners oftentimes train several machine learning models for a given problem, where each model differs in its architecture, training data composition, and/or hyperparameter settings. These models are then used to form an ensemble by combining the outputs of the various models, either by taking an average of outputs or learning weights to be associated with each model’s output.

> _“We … propose that under certain conditions one can average model parameters directly instead of retaining all networks and combine their outputs.”_ - from \[7\]

In the case of (simple) neural networks, we see in \[7\] that one can average model parameters directly instead of averaging model outputs. This approach yields similar performance to taking an average of each model’s output, while saving on both storage and compute costs. Most of the work we will see in this overview came long after \[7\], but this work served as a catalyst for model merging research, which—_as we will see_—became a fruitful and important topic of investigation.

#### (Linear) Mode Connectivity

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b901147-d5d5-4f1a-86f6-f7ddabea020f_1500x843.gif)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b901147-d5d5-4f1a-86f6-f7ddabea020f_1500x843.gif)

When we first learn about training a machine learning model via gradient-based optimization techniques (e.g., [stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)), we are typically shown a very simple, 1D view of a loss function being minimized; see above. In reality, the loss landscape of a neural network is non-convex and chaotic, as shown in the image below. However, there are some predictable properties and behaviors of these loss landscapes that we have observed empirically, which make them a little bit less intimidating. One of these interesting properties is _mode connectivity_.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce4f8ed7-b496-46fb-9903-0d3256a6a597_1074x568.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce4f8ed7-b496-46fb-9903-0d3256a6a597_1074x568.png)

([source](https://arxiv.org/abs/1712.09913))

**Mode connectivity** is an idea that was originally observed and coined in \[11\]. The training dynamics of neural networks are complex, as we see in the visualization above. For this reason, we might think that two independently-trained neural networks[3](https://cameronrwolfe.substack.com/p/model-merging#footnote-3-147448898) would end up in completely different regions of the optimization landscape. In \[11\], we learn that this is not always true—_the training trajectory of a neural network becomes relatively predictable after a certain number of iterations_.

> _“We show that the optima of these complex loss functions are in fact connected by simple curves over which training and test accuracy are nearly constant.”_ - from \[11\]

In particular, authors in \[11\] observe that the weights of neural networks that are trained independently can be connected together in the loss landscape via a path of constant training and test accuracy[4](https://cameronrwolfe.substack.com/p/model-merging#footnote-4-147448898), which can be discovered via their novel training procedure. Interestingly, these “modes” (i.e., the location of the trained network’s weights in the loss landscape) are usually connected by simple curves, as shown within the figure below. This idea was coined _mode connectivity_ due to our ability to connect the modes of these networks via simple paths of constant performance. This property was shown in \[11\] to hold for numerous computer vision architectures (primarily [ResNets](https://arxiv.org/abs/1512.03385)) trained on several popular datasets.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecc97147-77d2-467f-a033-2af740a64fad_1838x1028.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecc97147-77d2-467f-a033-2af740a64fad_1838x1028.png)

(from \[11\])

**Linear mode connectivity** is a more specific case of mode connectivity that is observed and analyzed in \[12\]. Authors in this paper study and compare the properties of vision models trained with different random noise. In particular, the data ordering and augmentation adopted during training, which is referred to as _SGD noise_ in \[12\], is varied. Then, the mode connectivity between the resulting models obtained via training with varying levels of SGD noise is studied.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0a59565-69ca-4067-9364-e67157e49d43_2174x750.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0a59565-69ca-4067-9364-e67157e49d43_2174x750.png)

(from \[12\])

To check if two networks are linearly mode connected, we just (linearly) [interpolate](https://en.wikipedia.org/wiki/Interpolation) between their weights and check that the training and test loss of models obtained along this path of linear interpolation is constant. Verifying linear mode connectivity is much simpler compared to mode connectivity in general, as we just have to check a linear path between the models’ weights, instead of using a more complex training algorithm to search for an arbitrary mode connected path. Going further, authors in \[12\] extend this analysis by only varying SGD noise after `k` training iterations have been performed; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6760e336-58ba-49a8-ad04-54b5ae4de5f0_2484x1262.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6760e336-58ba-49a8-ad04-54b5ae4de5f0_2484x1262.png)

(from \[12\])

We learn in \[12\] that neural networks become stable to SGD noise—_meaning that models trained with varying levels of SGD noise are still linearly mode connected after training_—very early in training. After a certain (reasonable) amount of training, _all networks obtained from the same base model are linearly mode connected_. As we will see, this finding is highly related to model merging, as we typically merge models by averaging or interpolating their weights. As such, linear mode connectivity provides empirical intuition for why such interpolated weights perform well!

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7279519a-c71e-433e-90ba-ffcc1647b117_1604x766.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7279519a-c71e-433e-90ba-ffcc1647b117_1604x766.png)

(from \[13\])

To make this idea a bit more specific, later work in \[13\] shows models that are finetuned from the same pretrained weights end up in the same basin (or region) of the loss landscape. In other words, independently finetuned models that start from the same base model end up close to each other in the parameter space—_this observation has deep connections to research on [critical learning periods](https://cameronrwolfe.substack.com/p/critical-learning-periods-in-deep-networks-35b2f17c4bbe)_. This region of the loss landscape contains several viable model parameters settings that can be discovered, which provides further explanation for the effectiveness of merging and interpolation techniques that will be explored throughout this overview.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71f5c6e8-217d-47dc-b48e-77f9d6374d64_2152x1002.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71f5c6e8-217d-47dc-b48e-77f9d6374d64_2152x1002.png)

(from \[14\])

**Mode connectivity and LLMs.** In \[14\], authors perform a similar analysis of linear mode connectivity for LLMs. From these experiments, we learn that finetuned LLMs tend to be linearly mode connected, but these models must be finetuned starting from the same pretrained weights for mode connectivity to hold! As shown above, numerous interpolation strategies are explored in \[14\] with GPT-style LLMs—_[GPT-2](https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2) in particular._ This work successfully demonstrates the applicability of linear mode connectivity to modern language models.

#### Pruning and Sparsity for Language Models

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe69db49f-4948-43ab-8d53-445cabdc4e1a_2052x666.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe69db49f-4948-43ab-8d53-445cabdc4e1a_2052x666.png)

(from \[15\])

Throughout this overview, we will study several works that explain the mechanisms that make model merging possible. At a high level, one important reason that model merging tends to be so effective is that large neural networks—_and LLMs in particular_—usually exhibit high levels of sparsity. The weights and activations within these models have a few important values, while other values are either redundant or not impactful; see above. As a result, we can eliminate a large ratio of model parameters via techniques like pruning without meaningfully impacting performance, as well as merge parameters without a high likelihood of two important weights or activations conflicting with each other.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8393f834-f04e-42a4-a25f-c84938fd658a_1400x384.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8393f834-f04e-42a4-a25f-c84938fd658a_1400x384.png)

Key components of neural network pruning

**Neural network pruning** starts with a large neural network and aims to derive from this larger network a smaller network with comparable performance. To do this, we usually follow a multi-step process (shown above) of:

1.  Training the model to convergence.
    
2.  Pruning the model’s weights (e.g., via a heuristic like removing the lowest-magnitude weights in each layer).
    
3.  (Optionally) rewinding the model’s remaining weights to their original values.
    
4.  Training the subnetwork (i.e., the pruned model) to convergence.
    

We can repeatedly apply steps two through four to derive an _iterative_ pruning strategy, which allows higher performance to be maintained by slowly removing weights from the model instead of pruning many weights at the same time. By following this strategy, we can derive incredibly small networks that achieve performance that is comparable to—_or even better than_—the larger models from which they are derived. For this reason, pruning became a popular research topic in the late 2010’s and continues to be an active direction of research even today.

> _“We articulate the lottery ticket hypothesis: dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that—when trained in isolation—reach test accuracy comparable to the original network in a similar number of iterations.”_ - from \[16\]

In a prior overview[5](https://cameronrwolfe.substack.com/p/model-merging#footnote-5-147448898), we learned about the topic of neural network pruning in great depth; see [here](https://cameronrwolfe.substack.com/p/saga-of-the-lottery-ticket-hypothesis-af30091f5cb). For those who want to learn more, I have also listed my favorite pruning papers below:

*   _[Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/abs/1608.03665)_:this paper is the first to explore pruning via the [L1-norm](https://mathworld.wolfram.com/L1-Norm.html) heuristic (i.e., removing low magnitude weights), which is the most commonly-used pruning strategy.
    
*   _[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)_: this paper shows significant reductions in the compute cost of a neural network can be achieved without a reduction in performance by removing filters with low L1-Norm from a model’s layers.
    
*   _[Rethinking the Value of Network Pruning](http://rethinking%20the%20value%20of%20network%20pruning/)_: this work performs an extensive empirical analysis of different pruning techniques and settings to determine best practices for obtaining high-performing subnetworks.
    
*   _[The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)_: this paper discovers and analyzes the lottery ticket hypothesis (LTH), which shows that high-performing subnetworks exist within the randomly-initialized weights of a neural network.
    

Despite becoming a very popular research topic in recent years, the idea of pruning has deep roots in early research on neural networks. Some of the first works to explore this idea were written in the early 1990s \[17, 18\]; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F49af9e0d-819e-4af5-8583-5726cebdd786_1868x736.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F49af9e0d-819e-4af5-8583-5726cebdd786_1868x736.png)

(from \[17, 18\])

**Pruning in the age of LLMs.** More recently, research on neural network pruning has been modernized in the age of LLMs. In \[19\], authors propose a pruning algorithm, called SparseGPT, that can be used to prune GPT-style language models—_using an unstructured approach_—to over 50% sparsity in one shot, meaning that no retraining is required after pruning. Eliminating retraining from the pruning procedure reduces compute costs significantly. The SparseGPT algorithm, which is shown below, operates by reducing the pruning process into a series of sparse regression problems that can be approximated efficiently.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4e6bfb65-0737-4381-b0ce-2cc798956761_1966x700.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4e6bfb65-0737-4381-b0ce-2cc798956761_1966x700.png)

(from \[19\])

Shortly after, authors in \[19\] performed analysis showing that magnitude-based pruning—_a popular and widely-used technique for pruning_—works poorly for LLMs. Although SparseGPT improves upon this technique, authors mention that this pruning algorithm takes around 4-5 hours to execute for a ~100B parameter LLM when running on a single GPU. In other words, _the SparseGPT algorithm is still computationally expensive even if it does not require retraining to be performed_.

> _“Considering the past success of magnitude pruning on smaller networks, this result suggests that LLMs, despite having 100 to 1000 times more parameters, are substantially more difficult to prune directly.”_ - from \[20\]

As a solution, Wanda (pruning by **W**eights **AND****A**ctivations)—_a very simple pruning approach for LLMs_—is proposed and analyzed in \[20\]. This approach determines which weights to prune by multiplying each of the model’s weights by their corresponding input activations on a per-output basis; see below for the exact formulation. Similar to SparseGPT, this technique requires no retraining. Additionally, Wanda is more efficient as a whole, and we can achieve higher levels of sparsity without damaging the performance of the pruned model. From this work, we learn that effectively pruning LLMs is difficult, but possible.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a4434c0-9d56-41df-b927-665521ff3f4e_2158x844.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a4434c0-9d56-41df-b927-665521ff3f4e_2158x844.png)

(from \[20\])

Although the technique used for pruning within Wanda might seem random, the algorithm in \[20\] is actually inspired by the recent observation that LLMs tend to have a small number of very high magnitude activations within each of their hidden layers. Interestingly, these high magnitude features (or outliers) seem to be an emergent property of larger models (i.e., ~7B parameters and beyond).

> _“At around 6.7B parameters, a phase shift occurs, and all transformer layers and 75% of all sequence dimensions are affected by extreme magnitude features.”_ - from \[21\]

This property was first observed in the quantization context \[21\], where authors develop a more performant 8-bit quantization technique for LLM inference by adopting tricks to property deal with these outliers features; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbff40d34-f893-4f19-899d-2ad115b1bef1_954x1252.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbff40d34-f893-4f19-899d-2ad115b1bef1_954x1252.png)

(from \[21\])

However, another work \[15\] also explores this property in depth, finding that these massive activations exist across a wide variety of different models and training settings. Put simply, work in \[15\] and \[21\] shows us that LLMs tend to have incredibly sparse activations that can be exploited for better pruning.

**Why does this matter?** Although pruning and model merging are separate topics, they are highly related due to their joint dependence upon the same fundamental idea—_sparsity_. Understanding pruning algorithms and related research is highly beneficial when learning about model merging, especially given that pruning is such a comprehensively-studied topic with an abundance of great ideas.

Early Work on Model Merging
---------------------------

Now that we understand the fundamental concepts that underlie the concept of model merging, we will look at a collection of notable papers that initially explored the concept of model merging for deep neural networks. Most of these papers were published prior to the popularization of LLMs, but many of the techniques are repurposed in the research we see on model merging today.

#### **[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482) \[5\]**

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb3f22f8-1801-4ad3-a128-5e319a2adf6a_1638x600.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb3f22f8-1801-4ad3-a128-5e319a2adf6a_1638x600.png)

(from \[5\])

When we finetune a pretrained model on some downstream task, the best training hyperparameters to use are oftentimes unknown. As a result, we usually _i)_ run several training trials with different hyperparameters, _ii)_ select the model that performs best on some held-out validation set, and _iii)_ discard the remaining models. Instead of discarding remaining models, we could form an ensemble of these models, which usually improves performance \[6\]. But, this approach drastically increases inference costs. In \[5\], authors explore an alternative strategy—_merging the weights of these models_—that can have the best of both worlds!

> _“Model soups can approach the performance of ensembling, with no additional computational cost or memory relative to a single model during inference.”_ \- from \[5\]

**Model soups.** The idea proposed in \[5\] is very simple—_we finetune several identical models with different hyperparameter settings and merge their weights_. Although we usually discard all but the best model when performing hyperparameter tuning, this approach is inspired by work on model ensembling that shows the benefit of averaging the predictions of these models. There are a few different ways we can merge the models’ weights! The following techniques are considered in \[5\]:

*   _Average_: just take a uniform average of models’ weights.
    
*   _Greedy_: select only models with performance above some threshold (e.g., average model performance) and take the average of selected models’ weights.
    

The resulting merged model is referred to as a “model soup”. Authors in \[5\] also propose a more sophisticated technique for _learning_ optimal merging coefficients between model weights. However, this approach requires all models to be loaded into memory at the same time, which is impractical.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e3ed84c-a7f0-4c99-ad8f-ac51abf4a4a5_1806x556.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e3ed84c-a7f0-4c99-ad8f-ac51abf4a4a5_1806x556.png)

When comparing average and greedy merging, we see in \[5\] that better performance is generally achieved with greedy merging. The issue with average merging is that certain hyperparameter settings may lead a subset of resulting models to perform poorly compared to others. Greedy merging removes these models from consideration via a simple performance filter; see above. More specifically, greedy soups are constructed by sequentially adding each model to the soup only if performance on a held-out validation set is improved. The list of considered model is sorted in decreasing order of validation set performance beforehand, _ensuring that the greedy soups is no worse than the best individual model_.

**How well does this work?** Experiments in \[5\] largely consider the task of image classification on ImageNet, but a variety of models are considered, including [CLIP](https://arxiv.org/abs/2103.00020), [ALIGN](https://arxiv.org/abs/2102.05918), [BASIC](https://arxiv.org/abs/2111.10050), and several [vision transformer (ViT)](https://cameronrwolfe.substack.com/p/vision-transformers) variants. These models are all pretrained on a separate large-scale dataset (e.g., [WIT](https://arxiv.org/abs/2103.01913) or [JFT-3B](https://paperswithcode.com/dataset/jft-3b)), finetuned on ImageNet, and evaluated on ImageNet[6](https://cameronrwolfe.substack.com/p/model-merging#footnote-6-147448898).

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcdd46868-e88c-4fc8-b0c2-0d98442c7f35_1564x544.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcdd46868-e88c-4fc8-b0c2-0d98442c7f35_1564x544.png)

(from \[5\])

The high level results of these experiments are outlined in the table above. For any number of models, the greedy soup consistently outperforms the best single model of any hyperparameter sweep, as well as nearly matches the performance of model ensembles in most cases. Unlike an ensemble, however, the model soup incurs no additional inference or memory cost relative to a single model!

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff5a245a2-2eca-41ff-8b46-49febda41465_1886x352.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff5a245a2-2eca-41ff-8b46-49febda41465_1886x352.png)

(from \[5\])

Authors in \[5\] even reach new state-of-the-art performance on ImageNet, achieving a test accuracy of 90.94%[7](https://cameronrwolfe.substack.com/p/model-merging#footnote-7-147448898) with a soup of [ViT-G](https://arxiv.org/abs/2106.04560) models. Model soups also perform better in the zero-shot regime and on new tasks that go beyond the distribution of the ImageNet test set. Additionally, model soups are found to yield useful results on text classification tasks with transformers; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7fe5747-27d1-44a7-b000-b8c453f88fe2_1970x1038.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7fe5747-27d1-44a7-b000-b8c453f88fe2_1970x1038.png)

(from \[5\])

**Why does this work?** Authors in \[5\] provide extensive analysis to motivate and explain the effectiveness of model soups. Much of this analysis draws parallels to prior research on linear mode connectivity. As shown in the figure above, finetuned models tend to lie within a similar basin of the loss landscape, meaning that the loss is relatively stable—_and might even decrease_—when we interpolate between the weights of these finetuned models! The best performing model is not any of the individual finetuned models. Rather, it lies somewhere in the loss landscape between all of these models. So, interpolating between these models via a model soup allows us to discover higher-performing models!

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63867b16-5afc-4b58-9bea-03f6ae7e17d1_2294x1196.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63867b16-5afc-4b58-9bea-03f6ae7e17d1_2294x1196.png)

(from \[36\])

**Model Ratatouille \[36\]** is an interesting extension to model soups that aims to repurpose the large variety of finetuned foundation models that are available online. Our goal is to finetune a model on a downstream task. Instead of directly finetuning a single model, however, we do the following (shown above):

1.  Obtain several openly-available models that have been finetuned on different auxiliary tasks.
    
2.  Finetune each of these models separately on our downstream task.
    
3.  Merge the weights (via an average) of all finetuned models.
    

We see in \[36\] that such a technique outperforms prior merging techniques, demonstrating that the extra information learned by these models when finetuned on diverse auxiliary tasks is useful for forming model soups.

#### [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903) \[8\]

> _“Can zero-shot \[foundation\] models be fine-tuned without reducing accuracy under distribution shift?”_ - from \[8\]

When we train a foundation model, we would like for this model to work well across a broad distribution of data. The advent of [self-supervised pretraining](https://cameronrwolfe.substack.com/i/139646437/language-model-pretraining) has largely solved this problem—_pretrained LLMs can solve a wide variety of problems in a zero-shot manner due to their sizable knowledge base that is derived from the massive text corpus on which the model is pretrained_. Still, we can improve the performance of a pretrained model on a particular target domain by finetuning the model on a targeted set of data from that domain[8](https://cameronrwolfe.substack.com/p/model-merging#footnote-8-147448898); see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F354c2e1c-c4f9-499f-9d14-76c18cc0f738_2464x846.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F354c2e1c-c4f9-499f-9d14-76c18cc0f738_2464x846.png)

Depiction of pretraining and finetuning

Despite the utility of finetuning, there are downsides of which we should be aware! Finetuning a pretrained model on a domain-specific dataset:

*   Improves the model’s performance on data from that particular domain.
    
*   Degrades the model’s performance on data beyond that particular domain.
    

Put simply, _finetuning makes the model less generic_, which can degrade its performance on data that is different from the finetuning dataset. In \[8\], authors try to mitigate this issue by adopting a simple model merging approach.

**Finetuning and robustness.** The negative impact of finetuning on the robustness of a pretrained model makes sense intuitively. Finetuning a pretrained model specializes this model to the properties of data from the target domain, which comes at the cost of model performance in a broader sense. Although the impact of finetuning on model robustness makes sense intuitively, authors in \[8\] analyze this relationship in more detail. In particular, several interesting findings regarding the impact of finetuning on model performance are outlined:

1.  The model’s performance improves on the target distribution.
    
2.  The model’s performance degrades under various distribution shifts (i.e., data that goes beyond the target distribution).
    
3.  Hyperparameter settings have a very large impact on robustness.
    
4.  More “aggressive” finetuning (e.g., using a larger learning rate) exacerbates these findings—_target domain performance improves more and performance is even worse under distribution shifts_.
    

To measure accuracy under distribution shifts, we can simply adopt existing datasets from [out-of-distribution (OOD) generalization research](https://arxiv.org/abs/2108.13624). For example, the ImageNet dataset has numerous alternative test sets that can be used to study several kinds of distribution shifts; see below for a few examples.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5f19b5bb-c9ef-4b87-ae00-101c6a44fbd8_1682x642.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5f19b5bb-c9ef-4b87-ae00-101c6a44fbd8_1682x642.png)

(from \[8\])

**Weight-Space Ensembles for Fine-Tuning (WiSE-FT).** The technique proposed in \[8\] is simple. We _i)_ start with a pretrained model, _ii)_ finetune the model on a target dataset, and _iii)_ interpolate between the weights of the pretrained and finetuned models; see below. Although we can arbitrarily interpolate between the weights of the pretrained and finetuned models, we see in \[8\] that simply taking an average of these weights works well in most cases.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F786d1c7c-5778-4096-be2c-a02b03eea745_1742x632.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F786d1c7c-5778-4096-be2c-a02b03eea745_1742x632.png)

Merging model weights in WiSE-FT (from \[8\])

Extending upon research on linear mode connectivity, authors in \[8\] observe that finetuned models are typically mode connected to their associated base model. More generally, we learn that models sharing a large part of their training trajectory—_such as a pretrained model and any finetuned models derived from this pretrained model_—tend to be mode connected, which allows us to merge these models without causing a catastrophic impact to performance.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a3190fa-4f8c-4d48-9921-49243d4b4890_1512x1292.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a3190fa-4f8c-4d48-9921-49243d4b4890_1512x1292.png)

(from \[8\])

**Does this work well?** The impact of WiSE-FT is mostly studied using pretrained [CLIP](https://arxiv.org/abs/2103.00020) models on the ImageNet dataset, where we see that merging pretrained and finetuned models yields a happy medium between the performance of both models. The results of the analysis in \[8\] are summarized in the figure above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fff692788-7829-447b-b127-bcbd3e08f793_1862x966.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fff692788-7829-447b-b127-bcbd3e08f793_1862x966.png)

(from \[8\])

In short, WiSE-FT yields the following benefits compared to a model obtained via standard finetuning (see table above for more details):

*   Improved accuracy under distribution shifts.
    
*   Comparable (or improved) accuracy in the target domain.
    

We also see that WiSE-FT can mitigate the sensitivity of model robustness to hyperparameter settings—_we can recover the performance of the best hyperparameter setting in nearly all cases by just changing the interpolation coefficient_[9](https://cameronrwolfe.substack.com/p/model-merging#footnote-9-147448898)_._ WiSE-FT also has no added computational costs during finetuning or inference.

**Why does this work well?** Beyond observing the performance benefits of WiSE-FT, authors dig a bit deeper into the mechanics of this technique by studying the relationship between the predictions generated by the pretrained, finetuned, and merged models. Interestingly, we see from this analysis that the finetuned model frequently overrides the predictions of the pretrained model when evaluating on in-domain data that is similar to the finetuning dataset. In contrast, predictions on out-of-distribution model are usually handled by the pretrained model! Put simply, the merged model naturally relies on the more appropriate model based upon the data (and task) being considered by a given input example.

> _“Overall, WiSE-FT is simple, universally applicable in the problems we studied, and can be implemented in a few lines of code. Hence we encourage its adoption for fine-tuning zero-shot models.”_ - from \[8\]

**Implementation details.** Given that WiSE-FT just takes a weighted average of model parameters, this technique is actually very easy to implement! An example using PyTorch syntax is provided in the algorithm below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc10a4bb9-eb0a-4a23-93dc-6421a7e28e91_792x746.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc10a4bb9-eb0a-4a23-93dc-6421a7e28e91_792x746.png)

(from \[8\])

**Further research.** WiSE-FT is further analyzed in \[42\], where we see that this method also improves generalization via the “FalseFalseTrue” phenomenon. More specifically, WiSE-FT is observed to correct numerous cases where each model makes an incorrect prediction. The merged model is correct, but both of the source models are wrong! After analyzing this property theoretically, authors conclude that this property is largely due to impact of diverse feature sets on OOD generalization, thus providing (for the first time!) theoretical intuition for the ability of weight-space ensembles to outperform output-space ensembles.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F220060ac-0d9b-4a4c-98a8-760079c94ec4_2140x782.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F220060ac-0d9b-4a4c-98a8-760079c94ec4_2140x782.png)

(from \[49\])

An extension of WiSE-FT for LLMs, called an LM-Cocktail, is proposed in \[49\]. This approach merges finetuned language models with their pretrained base models, which mimics the strategy used in WiSE-FT. However, the technique is slightly more general, as additional models—_such as those finetuned on data from other domains (i.e., peer models)_—can also be included in the merge; see above.

#### **[Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522) \[9\]**

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7a7ce1a5-74c0-4ab6-9556-f661848e0e18_1404x984.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7a7ce1a5-74c0-4ab6-9556-f661848e0e18_1404x984.png)

(from \[9\])

Authors in \[8\] present a more recent extension of the model soups \[5\] and WiSE-FT \[8\] papers that aims to find a better tradeoff between:

*   The performance of the merged model.
    
*   The number of finetuned models we need to train (and merge).
    

Beginning with a pretrained model, the model soups technique proposed in \[5\] mandates that we perform several, independent finetuning runs, then take an average of the resulting models’ weights. Unfortunately, this technique typically requires that we finetune dozens of models, which is computationally expensive! Instead, WiSE-FT proposes finetuning a single model and interpolating between the weights of this model and the pretrained model, but performance is lacking.

> _“This strategy can be aptly coined Model Stock, highlighting its reliance on selecting a minimal number of models to draw a more optimized-averaged model.”_ \- from \[9\]

To solve these issues, authors in \[9\] deeply analyze the geometric properties of finetuned weights relative to a pretrained model. This analysis allows them to devise an efficient merging strategy, called a model “stock”, that can achieve performance comparable to a model soup with only two finetuning runs, thus saving a massive amount of computation in terms of total training costs.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F670a222e-8535-47fe-9011-12b4abac4d05_1404x728.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F670a222e-8535-47fe-9011-12b4abac4d05_1404x728.png)

(from \[9\])

**Geometric analysis.** To arrive at this efficient merging strategy, we need to understand the properties of finetuned weights. _What are model soups and WiSE-FT actually doing, and why do they work well?_ To answer these questions, authors in \[9\] finetune over 50 pretrained CLIP models and study how the finetuned weights relate to each other[10](https://cameronrwolfe.substack.com/p/model-merging#footnote-10-147448898), arriving at an interesting observation (depicted above):

1.  All finetuned weights lie on a thin shell (or sphere) that is centered around a central point (i.e., the average of the finetuned weights), meaning the distance between finetuned weights and this center is (roughly) constant.
    
2.  The center point lies at a different location in space relative to the pretrained weights, but these positions still satisfy predictable geometric properties.
    

This observation is empirically validated in \[9\] using several different finetuning setups, models, and datasets. Furthermore, we see from the analysis in \[9\] that these “central” weights—_those lying at the center of all finetuned weights_—consistently achieve optimal performance, exceeding the performance of all finetuned models and the pretrained model. This finding explains the utility of model soups—_the model soup is an average of finetuned models and, therefore, an approximation of these central weights that are found to perform well_.

> _“We uncover a strong link between the performance and proximity to the center of the weight space \[and\] introduce a method that approximates a center-close weight using only two finetuned models.”_ - from \[9\]

With this analysis in mind, WiSE-FT can be viewed as simply interpolating between the weights of a finetuned and pretrained model, which can be used to discover a point in space that is closer to the high-performing center; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc4d0c6a3-6749-4077-abc5-3a9ff7e6bd16_1412x528.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc4d0c6a3-6749-4077-abc5-3a9ff7e6bd16_1412x528.png)

(from \[9\])

**How can we use this?** The analysis performed in \[9\] is incredibly thorough and interesting, but the details are intricate and beyond the scope of this post. I’d highly encourage the interested reader to check out sections two and three of the paper for the full details. The main question we want to answer here is: _How can we practically leverage this information to create a better model merging technique?_

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecdf66c8-909f-4056-92b4-f6973abc76f4_1416x812.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecdf66c8-909f-4056-92b4-f6973abc76f4_1416x812.png)

(from \[9\])

The high-level answer is simple—_we can use the geometric relationship between pretrained and finetuned model weights to efficiently approximate the center point_. This approximation, which uses the properties discovered in \[9\] to directly solve for the center point, only requires two finetuned models to be generated; see above. The pretrained model serves as an “anchor point”, and we can approximate the center point by projecting it onto the plane formed by the weights of all three models.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcf8b444b-92cf-44af-9282-910521cc1702_1302x792.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcf8b444b-92cf-44af-9282-910521cc1702_1302x792.png)

Creating a model stock (from \[9\])

There is a lot of fancy math here that can be a bit difficult to understand, _but the practical result of this extensive analysis is just a different equation (shown above) for finding the best interpolation coefficient between finetuned weights_! So, this approach is not actually that much different than WiSE-FT at the end of the day, we just _i)_ train two finetuned models instead of one and _ii)_ use a more intricate technique—_based upon the geoemtric analysis in \[9\]_—to optimally merge the models. However, WiSE-FT is more widely adopted in the literature due to its simplicity.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08957412-ea76-4455-b4b6-1e35f023b40a_1408x812.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08957412-ea76-4455-b4b6-1e35f023b40a_1408x812.png)

(from \[9\])

**Empirical results.** We see in \[9\] that closer proximity to the center point formed by the average of finetuned model weights—_assuming the average is taken over a sufficiently large number of finetuned models_—yields better performance, both on the target domain and under a distribution shift. Most experiments in \[9\] are performed using [pretrained CLIP](https://arxiv.org/abs/2103.00020) models that are finetuned on ImageNet, thus matching the experimental setup of WiSE-FT \[8\]. Model stocks with only two finetuned models are found to reach state-of-the-art performance on ImageNet and match the performance of model soups \[5\]—_obtained using dozens of finetuned models_—in this regime, thus yielding a significant reduction in training costs; see above. Interestingly, adding more models (i.e., three or four finetuned models) to a model stock does not yield a significant performance benefit; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0b816d39-d48c-4312-aa9c-73055a305f8b_782x608.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0b816d39-d48c-4312-aa9c-73055a305f8b_782x608.png)

#### Weight Averaging Techniques

> _“We show that SGD generally converges to a point near the boundary of the wide flat region of optimal points. SWA is able to find a point centered in this region, often with slightly worse train loss but substantially better test error.”_ - from \[22\]

One common variant of model merging is simply taking an average of model weights at several points throughout the model’s training trajectory. More specifically, we _i)_ record several checkpoints of the model’s weights throughout training, _ii)_ take an average of the model’s weights at several checkpoints, and _iii)_ use these averaged weights as our final model. This technique was originally proposed in \[22\][11](https://cameronrwolfe.substack.com/p/model-merging#footnote-11-147448898) and is referred to as _stochastic weight averaging (SWA)_; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb83fe23f-591e-4d19-bc2a-3385b3498a6e_2276x984.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb83fe23f-591e-4d19-bc2a-3385b3498a6e_2276x984.png)

(from \[22\])

Compared to standard training (with stochastic gradient descent), SWA is shown in \[22\] to help the resulting model to generalize better. Plus, this technique is easy to implement and has no computational overhead, whereas alternative techniques (e.g., creating an ensemble of models) have significant increases in computation at inference time. We start with a pretrained network and apply SWA during the finetuning process, which ensures that the checkpoints are mode connected \[13\]. Numerous other weight averaging techniques have been explored as well:

*   [DiWA](https://arxiv.org/abs/2205.09739) \[35\] extends upon prior weight averaging techniques by improving the diversity of the models being averaged.
    
*   [SWAD](https://arxiv.org/abs/2102.08604) \[41\] extends SWA via a stochastic weight sampling strategy that finds flatter minima—_with better generalization properties_—in the loss landscape.
    
*   [Fuse to Forget](https://arxiv.org/abs/2311.07682) \[43\] studies the properties of knowledge within averaged models, finding that the resulting models tend to _i)_ lose unshared knowledge and _ii)_ have their shared knowledge enhanced.
    
*   Authors in \[44\] show that weight averaging can be used as a mechanism to mitigate catastrophic forgetting of old tasks when automatic speech recognition (ASR) models are adapted to new tasks.
    
*   We see in \[45\] that weight averaging is useful for merging learned policies—_implemented as a “decision” transformer_—for locomotion within reinforcement learning (RL), while \[46\] shows us that weight averaging is generally useful for improving training stability for RL with deep neural networks.
    

**EMA of weights.** Instead of taking an average over a finite and discrete number of model checkpoints, we can take an [exponentially moving average (EMA)](https://leimao.github.io/blog/Exponential-Moving-Average/) of model weights throughout the training process. This technique, which is an extension of SWA, was heavily adopted by vision models in the late 2010s (e.g., [InceptionNet](https://arxiv.org/abs/1512.00567), [EfficientNet](https://arxiv.org/abs/1905.11946), [MnasNet](https://arxiv.org/abs/1807.11626), and more) but is not commonly covered in papers—_it’s more of a practical implementation detail that can be found in the code repositories for these models_. However, the concept of using EMA during training is mentioned in the original Adam optimizer paper \[24\]; see Section 7.2 [here](https://arxiv.org/abs/1412.6980).

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F87500599-7a5f-4809-b629-92245ca1b464_1432x960.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F87500599-7a5f-4809-b629-92245ca1b464_1432x960.png)

(from \[26\])

The idea of using models—_or predictions_—obtained via EMA as a target for self or semi-supervised learning has also been explored a lot \[25, 26\]; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46b2f4fa-240d-4548-b6cf-81a2067ae97c_834x888.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46b2f4fa-240d-4548-b6cf-81a2067ae97c_834x888.png)

**Extension to LLMs.** In \[27\], authors explore an extension of SWA for LLM pretraining, leading to faster convergence and improved generalization. Two key observations inspire the proposed technique in \[27\]:

1.  Models trained with a larger learning rate see more of a benefit from averaging model checkpoints along the training trajectory.
    
2.  Averaging model trajectories that are further apart in the training process leads to larger gains.
    

Based on these findings, authors propose LAWA (**LA**test **W**eight **A**veraging), which performs checkpoint averaging using a sliding window along the training trajectory of an LLM; see above for an illustration. LAWA simply maintains a first-in-first-out buffer of checkpoints sampled throughout the training process and computes an average over the `k` most recent checkpoints in this buffer. Checkpoints are inserted into the buffer with many training steps in between, ensuring that more distant checkpoints (i.e., those from the earlier phases of training) are included within the merging process; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2631968c-c1e9-4feb-adfc-e09b66f00e36_2148x374.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2631968c-c1e9-4feb-adfc-e09b66f00e36_2148x374.png)

(from \[27\])

In \[27\], we see that LAWA outperforms conventional checkpoint averaging techniques like EMA and SWA in the language modeling domain, where larger learning rates are usually preferred. LLMs trained with LAWA converge faster and generalize better. The observed benefits of LAWA tend to improve as we add more training steps between the checkpoints that are sampled for averaging.

Model Merging for Large Language Models
---------------------------------------

Now that we have learned about early work on model merging, we will take a look at more recent techniques, such as task vectors, TIES, DARE, and more. Those who have tracked recent developments in LLM research are likely to have seen these techniques. Many of these algorithms are supported within popular open-source software for LLMs, such as the [mergekit](https://huggingface.co/blog/mlabonne/merge-models) \[54\]. In this section, we will take a look at the most common algorithms used for merging LLMs, as well as learn about how model merging has transformed the LLM alignment process.

#### **[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) \[1\]**

> _“With task arithmetic, practitioners can reuse or transfer knowledge from models they create, or from the multitude of publicly available models all without requiring access to data or additional training.”_ - from \[1\]

As practitioners, we usually solve tasks by _i)_ starting with a pretrained model (e.g., LLaMA-3 or Mistral) and _ii)_ adapting (or [steering](https://x.com/cwolferesearch/status/1645535869556727815)) this model to our desired use case. For example, we might further finetune the model on a downstream task, try to reduce the model’s bias, or perform [alignment](https://cameronrwolfe.substack.com/i/138218863/language-model-alignment). To do this, we have two basic options at our disposal:

*   Specify desired behavior as an instruction within the model’s prompt (i.e., use [prompt engineering](https://cameronrwolfe.substack.com/p/modern-advances-in-prompt-engineering)).
    
*   Finetune the model on extra data.
    

Usually, we will try to solve a problem via prompting first due to simplicity, then perform finetuning if performance is short of what we need or expect. However, the process of finetuning an LLM requires task-specific data and can be expensive, both in terms of time and monetary costs. In \[1\], authors propose a much simpler and easier way of editing pretrained models, called task arithmetic.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc04793ca-23ad-468a-991f-7ff4848c45e4_714x818.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc04793ca-23ad-468a-991f-7ff4848c45e4_714x818.png)

Depiction of a task vector (from \[1\])

**What is a task vector?** The first concept introduced in \[1\] is a task vector, which simply refers to a vector that is obtained by subtracting a finetuned model’s weights from a pretrained model; see above. Intuitively, a task vector encodes all of the information needed to solve a task that is learned via finetuning. These task vectors live within the parameter space of a neural network, meaning that they are the same size and shape as the weights of the model we are trying to edit.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F68818efb-b617-4479-a69b-32805edb6095_624x684.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F68818efb-b617-4479-a69b-32805edb6095_624x684.png)

In \[1\], we see that these task vectors can be used to change the behavior of a pretrained model—_task vectors are a simple and cheap alternative to prompting or finetuning_. In particular, we can edit a model by performing arithmetic between the model’s parameters and a task vector, as shown above. The scaling term is typically tuned via a hold-out validation set. Intuitively, these task vectors move the parameters of our model towards those of a model with the desired behavior. This approach requires that all models share the exact same architecture.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7bda1c4b-4f57-48a8-b03d-c1752c6eaa94_1348x728.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7bda1c4b-4f57-48a8-b03d-c1752c6eaa94_1348x728.png)

**Types of arithmetic.** Beyond basic addition, we see in \[1\] that there are several forms of meaningful arithmetic that can be performed with task vectors; see above. We can add a task vector to learn a new skill or negate a task vector to eliminate a skill. _We can even add or negate several task vectors at once!_

> _“Negating a vector can be used to remove undesirable behaviors or unlearn tasks, while adding task vectors leads to better multi-task models, or even improves performance on a single task.”_ - from \[1\]

Going further, authors find in \[1\] that analogies between task vectors hold as well. Assume we have four tasks A, B, C, and D with a analogous relationship given by _“A is to B as C is to D”_. Then, we can improve performance on task D using the task vector shown below. Here, we construct a task vector by:

*   Finding the difference in task vectors between tasks A and B.
    
*   Adding this difference to the task vector for task C.
    

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe7dcb89d-5ce9-4a8c-8a6a-0565b4b6d442_434x302.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe7dcb89d-5ce9-4a8c-8a6a-0565b4b6d442_434x302.png)

Analogy-based task vector (from \[1\])

**Experimental results.** Task vectors are applied to numerous types of models in \[1\], including both LLMs and specialized models (e.g., image and text classifiers). We see that adding and negating task vectors is clearly effective. For example, we can obtain a toxicity task vector by finetuning an LLM on toxic text. Then, we can make an LLM less toxic by negating this task vector; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0a327f0a-a3b6-4f1f-b1ac-876b39158744_1236x512.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0a327f0a-a3b6-4f1f-b1ac-876b39158744_1236x512.png)

(from \[1\])

We can also forget certain image classification tasks via negation, as well as learn new tasks—_even multiple tasks at the same time_—via addition. For example, the image below show how pairs of task vectors can improve an image classifier’s performance on downstream tasks, while the table shows that tasks vectors obtained from an LLM that has been finetuned on a downstream task can be used to improve a model’s performance on that task without extra finetuning!

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5faea64-7570-4783-8622-efd58669f0e1_864x752.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5faea64-7570-4783-8622-efd58669f0e1_864x752.png)

(from \[1\])

Additionally, these improvements in performance seem to avoid associated degradations in performance on control tasks—_the model does not get worse in other areas as a result of the task vector_. Similar experimental results are observed for analogous task vectors, where we see that most task vectors are [orthogonal](https://en.wikipedia.org/wiki/Orthogonality) and can be used to generalize to new domains. In many ways, this analysis is reminiscent of early analysis of [word vectors](https://arxiv.org/abs/1301.3781), where we observed similar analogous relationships between the vectors of associated words; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0e8007ac-2c20-4dd1-abb7-f1bcd5c506a5_604x436.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0e8007ac-2c20-4dd1-abb7-f1bcd5c506a5_604x436.png)

([source](https://arxiv.org/abs/1810.04882))

**Key benefits.** Compared to prompting or finetuning, editing models via task arithmetic is cheap and easy. We do not need any external data or GPUs for training. Rather, we just need access to a finetuned model—_many of which are already available online_! Task arithmetic only requires element-wise operations on the model’s weights to quickly experiment with different task vectors. So, we can easily re-use or transfer knowledge from models that are openly available.

#### [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) \[2\]

> _“When merging a parameter that is influential for one model but redundant (i.e. not influential) for other models, the influential value may be obscured by the redundant values, lowering the overall model performance.”_ - from \[2\]

Given the proliferation of task-specific, finetuned models online, model merging is a promising technique that could help to consolidate these models. However, the performance of basic model merging techniques (e.g., averaging or weighted averaging of parameters) tends to degrade as we merge larger numbers of models. In \[2\], authors study the task vector-based model merging regime \[1\] and identify that performance degradations due to model merging are largely caused by “interference” that occurs between model parameters during the merging process. In particular, two key sources of interference are identified:

1.  _Redundant parameters_: many parameters in a task vector are redundant, and removing these parameters does not impact performance.
    
2.  _Sign disagreements_: certain parameters may have a positive value for some models and negative value for others, which causes a conflict.
    

A schematic depiction of these different interference patterns is provided below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd85c544f-3192-4d90-80a2-ed4d7843caf5_980x896.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd85c544f-3192-4d90-80a2-ed4d7843caf5_980x896.png)

(from \[2\])

To prove that this interference is occurring, we can just study basic properties of model parameters. In the (left) figure below, we first see that model performance is largely determined by a small group of (high magnitude) parameters, while most other parameters are redundant—_meaning that they do not impact the model’s performance when removed_[12](https://cameronrwolfe.substack.com/p/model-merging#footnote-12-147448898). Similarly, we see that sign conflicts are very common and become more common as the number of models considered is increased.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73cf7b3b-082e-441b-be53-e0f640824acb_1350x542.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73cf7b3b-082e-441b-be53-e0f640824acb_1350x542.png)

(from \[2\])

**Dealing with interference.** The method devised in \[2\] for mitigating interference, called **T**r**I**m, **E**lect **S**ign, and Merge (TIES-Merging), adds three additional steps to the task vector-based model merging process:

*   _Trim_: retain only influential weights (i.e., the top `K`% of highest-magnitude values) in each task vector and set others to zero.
    
*   _Elect Sign_: choose the sign of total highest magnitude for each element across task vectors—_resulting in a sign vector_[13](https://cameronrwolfe.substack.com/p/model-merging#footnote-13-147448898)_—_by taking a element-wise sum across task vectors and seeing whether each resulting element is positive or negative.
    
*   _Disjoint Merge_: take the average of task vector values that agree with the majority sign, thus ignoring parameters that are either trimmed or have a sign conflict.
    

Once we have completed these three additional steps, we can perform model merging normally with the resulting task vector. The three steps within TIES-Merging are depicted below. Interestingly, we see in \[2\] that maintaining only the top 20% of task vector components yields stable performance—_indicating that a large majority of task vector components are redundant!_

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd03017b-007a-4a13-9133-ef929fdfc04a_1344x766.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd03017b-007a-4a13-9133-ef929fdfc04a_1344x766.png)

(from \[2\])

**Less interference is beneficial.** When TIES-merging is analyzed empirically, there are a variety of interesting findings from which we can learn. Overall, we see that TIES merging yields a clear benefit across a variety of experimental seeings; see below. In particular, TIES works well for multiple modalities (text and vision) and is even compatible with [parameter-efficient finetuning](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft).

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8175006e-b17f-4927-bd6a-bf0d905aa359_1186x488.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8175006e-b17f-4927-bd6a-bf0d905aa359_1186x488.png)

(from \[2\])

Compared to baseline techniques, TIES-merging is also found to generalize better to new tasks and have improved scaling properties as the number of models being merged is (reasonably) increased; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F159c85e4-68ad-430a-92d3-c55ca544e2de_1292x1050.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F159c85e4-68ad-430a-92d3-c55ca544e2de_1292x1050.png)

(from \[2\])

Both components of TIES-Merging—_removing redundant parameter and electing a majority sign_—are important, but properly estimating the majority sign seems to be especially important; e.g., flipping the sign of high magnitude parameters drastically deteriorates performance. Additionally, authors include interesting experiments in \[2\] that craft an [oracle](https://en.wikipedia.org/wiki/Oracle_machine) for estimating the majority sign by taking the sign of a model trained in a multi-task fashion. Using this sign oracle during the election process of TIES-Merging is actually found to improve performance!

**Further finetuning.** To go beyond the traditional model merging setup, we can perform extra finetuning with a model that is obtained via merging. In this domain, TIES-merging is shown in \[2\] to provide us with a better starting point. In particular, models obtained via TIES-Merging outperform those obtained via baseline merging techniques after further finetuning; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc8a7edf7-58ca-4bfc-96b4-042df2151ac7_594x406.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc8a7edf7-58ca-4bfc-96b4-042df2151ac7_594x406.png)

(from \[2\])

#### **[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099) \[3\]**

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa7e5fa3a-95b3-4e07-b7f9-d51700a754db_1648x1250.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa7e5fa3a-95b3-4e07-b7f9-d51700a754db_1648x1250.png)

(from \[3\])

> _“In XMen’s Apocalypse, the character can absorb the powers of other mutants to strengthen himself… the protagonist in Super Mario can gain superpowers by absorbing in-game items… we astonishingly find that language models (LMs) can enhance their capabilities by absorbing other models without retraining or even GPUs.”_ - from \[3\]

Authors in \[3\] propose an addition to existing model merging methods that is especially effective for language models that have underwent [supervised finetuning (SFT)](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised). When studying the different in parameters values between the base model and the model obtained after SFT—_referred to as “delta parameters”_—we see (again) in \[3\] that these parameter values have a lot of redundancy. As a result, many of these delta parameters can be eliminated via a technique proposed in \[3\], called **D**rop **A**nd **RE**scale (DARE). The DARE process makes language models finetuned with SFT much more amenable to model merging.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F18082791-5848-441a-8694-d3ad66f5c039_704x914.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F18082791-5848-441a-8694-d3ad66f5c039_704x914.png)

(from \[3\])

**What is DARE?** The concept proposed in \[3\] is actually very simple. We just:

1.  Randomly drop delta parameters (with probability `p`).
    
2.  Rescale remaining parameters by a factor of `1 / (1 - p)`.
    
3.  Add the remaining pruned and scaled parameters to the weights of the pretrained base model.
    

These steps are outlined in the figure above. Notably, _DARE is not a model merging technique_. Rather, it is a technique for sparsifying delta parameters within an SFT model that is found empirically to have minimal impact on the resulting model’s performance. In fact, we can even use DARE to eliminate up to 99% of delta parameters for sufficiently large language models; see below. Such a finding demonstrates that delta parameters for SFT models are highly redundant.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbed1bde3-5c94-44a9-8e63-79b4e6d9a851_962x520.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbed1bde3-5c94-44a9-8e63-79b4e6d9a851_962x520.png)

(from \[3\])

**Application to model merging.** Although DARE itself is not a model merging technique, it is a useful plug-in for existing methods (e.g., TIES-Merging \[2\]). The delta parameters considered by DARE are identical to the task vectors considered by prior model merging techniques \[1\]. DARE sparsifies these task vectors without damaging the underlying model’s performance, _which mitigates interference when the parameters of these models are actually merged_! Assuming models being merged share the same backbone, the chance of interference is much lower when merging multiple models to which DARE has been applied—_many of the delta parameters within these models will have been set to zero_.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd6b0041a-8a33-4bb2-b2c2-f32d4f271727_1656x930.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd6b0041a-8a33-4bb2-b2c2-f32d4f271727_1656x930.png)

(from \[3\])

In \[3\], authors merge various flavors of language models—_including both [encoder-only](https://cameronrwolfe.substack.com/i/76273144/transformer-encoders) and [decoder-only](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) models_—with and without DARE. In the table above, we see that using DARE on top of existing model merging techniques for decoder-only LLMs tends to (slightly) improve the performance of the merged model. This difference in performance is more noticeable when merging several specialized models (e.g., instruction-following, math, and code models).

To demonstrate its utility, DARE is used to create two 7B parameter LLMs by merging the [NeuralBeagle](https://huggingface.co/mlabonne/NeuralBeagle14-7B) / [Turdus](https://huggingface.co/udkai/Turdus) (`supermario-v1`) and [WildMorcoroni](https://huggingface.co/BarryFutureman/WildMarcoroni-Variant1-7B) / [WestSeverus](https://huggingface.co/PetroGPT/WestSeverus-7B-DPO-v2) (`supermario-v2`) models. These models achieve top performance (at the time) among 7B models on the [Open LLM leaderboard](https://huggingface.co/open-llm-leaderboard); see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F773b9392-bec8-4a88-b4a0-3c41e3636298_1122x426.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F773b9392-bec8-4a88-b4a0-3c41e3636298_1122x426.png)

(from \[2\])

The performance impact of DARE is even more noticeable for encoder-only models; see below (left). The performance of several specialized models can be maintained after merging even while keeping only a small number of delta parameters from each SFT model. However, the rescaling step of DARE is essential to performance under higher levels of sparsity; see below (right).

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8f6cd8a4-8ea2-4626-ae2e-00fb59012ecd_1436x460.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8f6cd8a4-8ea2-4626-ae2e-00fb59012ecd_1436x460.png)

(from \[3\])

**Can we always use DARE?** DARE is a sparsification technique that provides useful insight about language models (i.e., delta parameters are very sparse) and can be used to yield a slight boost in model merging performance. However, _DARE is not applicable to models in every setting_. We see in \[3\] that language models finetuned via SFT have uniquely small delta parameters—_minimal modifications are made to the pretrained LLM_. When similar models are finetuned using a [continued pretraining setup](https://arxiv.org/abs/2407.07263), we observe delta parameters with much larger magnitudes. As a result, applying DARE in this setting is more damaging to performance, especially when dropping larger ratios of delta parameters.

> _“This finding further confirms that SFT primarily unlocks the abilities of pre-trained LMs, rather than introducing new capabilities.”_ - from \[3\]

This finding has connections to prior analysis of the [Superficial Alignment Hypothesis](https://cameronrwolfe.substack.com/i/134561977/lima-less-is-more-for-alignment) \[4\], which posits that:

*   All of a language model’s knowledge is learned during pretraining.
    
*   Alignment serves the purposes of teaching the model how to properly surface this knowledge (e.g., style, format, tone, etc.).
    
*   Alignment can be highly data-efficient because of this.
    

With this in mind, we should not be too surprised that language models finetuned with SFT—_an alignment technique_—show a relatively small delta compared to the pretrained model. Similarly, continued pretraining should have a larger delta, as it usually serves the purpose of injecting new knowledge into the model.

#### **[WARP: On the Benefits of Weight Averaged Rewarded Policies](https://arxiv.org/abs/2406.16768) \[10\]**

> _“While weight averaging was initially mostly used for discriminative tasks… it is now becoming popular for generative tasks; its use in KL-constrained RLHF has already shown preliminary successes.”_ \- from \[10\]

As we have seen, model merging has a long history of interesting applications and techniques within deep learning. Recently, however, we have begun to see model merging explored in the context of [LLM alignment](https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-imitation). The success of model merging in this domain has had a noticeable impact on pipelines used for training frontier models—_merging is becoming a commonly-used component_.

**Refresher on alignment.** Most LLMs are trained using a three-stage process that includes [pretraining](https://cameronrwolfe.substack.com/i/136638774/language-model-pretraining), [supervised finetuning (SFT)](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised) and [reinforcement learning from human feedback (RLHF)](https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations); see below. During pretraining, we train the language model—_using [next token prediction](https://cameronrwolfe.substack.com/i/136638774/understanding-next-token-prediction)_—over large amounts of unlabeled text to build a large base of knowledge within the model. From here, we perform SFT and/or RLHF. These algorithms power the LLM finetuning (or alignment) process and are less computationally expensive relative to pretraining.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a07a22d-e7d4-4e78-895f-7cb957adc0d5_1978x902.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a07a22d-e7d4-4e78-895f-7cb957adc0d5_1978x902.png)

Steps of training and aligning a language model

Typically, we first perform SFT over a (relatively) small[14](https://cameronrwolfe.substack.com/p/model-merging#footnote-14-147448898), high-quality set of examples, providing us with a better “starting point” for RLHF. Then, we apply RLHF in an iterative fashion by continually collecting new batches of preference data and further finetuning the model. The purpose of alignment is not to instill new knowledge within the LLM, but rather to teach the model how to surface its existing knowledge to human users in a preferable manner.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa333055d-b628-4bfd-8afb-0a9b3b2f80ae_1806x482.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa333055d-b628-4bfd-8afb-0a9b3b2f80ae_1806x482.png)

RLHF objective with KL divergence (from \[10\])

**Why do we need model merging?** When using RLHF, we usually add a [Kullback-Leibler (KL) divergence](https://dibyaghosh.com/blog/probability/kldivergence.html) term (shown above) to the objective being used, which measures the distance between the current model and the SFT model—_or some other anchor model_. At a high level, this divergence term captures how much the model has changed during training with RLHF. By making sure this term does not become too large, we can avoid issues like:

*   Forgetting of knowledge from pretraining (i.e., the alignment tax \[28\]).
    
*   Reward hacking (e.g., verbose, unsafe, or flawed outputs).
    
*   Decline in the diversity of model outputs \[29\].
    

However, adding this divergence term also hinders the reward optimization, _forming a tradeoff between the model’s final reward and the KL divergence_. As outlined in \[10\], model merging is an effective technique for finding better tradeoffs between the KL divergence and reward during finetuning with RLHF!

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F11dfa2ba-897c-47ad-a3bd-85b977c1d5e2_2074x650.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F11dfa2ba-897c-47ad-a3bd-85b977c1d5e2_2074x650.png)

(from \[10\])

**Weight Averaged Rewarded Policies (WARP)** \[10\], depicted in the figure above, incorporates three types of model merging at three different phases of the LLM alignment process:

1.  We use an **Exponential Moving Average (EMA)** of model weights as an anchor for the KL divergence during finetuning with RLHF.
    
2.  We independently finetune `M` models via RLHF, then merge the policies using task vectors and **spherical linear interpolation (SLERP)**; see [here](https://en.wikipedia.org/wiki/Slerp) and above.
    
3.  We **linearly interpolate** the result towards the (SFT) initialization.
    

SLERP \[37\] can also be plugged in to any of the other model merging techniques we have seen so far as an alternative to linear interpolation.

> _“Merging task vectors, either with SLERP or LERP, combines their abilities. The difference is that SLERP preserves their norms, reaching higher rewards than the base models.”_ - from \[10\]

This multi-stage merging approach—_see below for a full outline of each stage_—uses several of the model merging techniques we’ve seen so far, including EMA \[24, 27\], weight averaging \[22\], task vectors \[1\], and WiSE-FT \[8\].

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb06a7ae-0126-4162-8130-a4af6359e347_1912x996.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb06a7ae-0126-4162-8130-a4af6359e347_1912x996.png)

Three different merging steps in WARP (from \[10\])

WARP has additional training costs—_due to the need to independently train several models via RLHF_—but has no additional memory or compute requirements at inference time. Model merging yields a unique benefit at each stage of the WARP algorithm. Using the EMA of model weights as an anchor for KL divergence allows the divergence constraint to be relaxed over time—_weights are tied to the SFT model early on and gradient updates become more aggressive at the end of training_.

Similarly, using SLERP to merge the weights of several RLHF models yields an improvement in reward, while interpolating back towards the SFT initialization finds a better tradeoff between reward and KL divergence. WARP is applied in an iterative fashion by performing several “rounds” of finetuning, where the final model at each iteration is used as an initialization for the next. This iterative approach matches the usual strategy used for RLHF within the literature.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d61245d-ba11-4a4e-b758-fdc1b941d68c_1268x1212.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d61245d-ba11-4a4e-b758-fdc1b941d68c_1268x1212.png)

(fr10\])

**Using WARP in practice.** The result of this combination of merging techniques is a model that achieves the best possible tradeoff between KL and reward. The benefits of WARP are empirically validated in \[10\] using Gemma \[30\]. Models are evaluated in terms of _i)_ their final reward and _ii)_ KL divergence with respect to the SFT initialization. From experiments, we see that WARP achieves a better reward-KL tradeoff compared to other RL-based alignment techniques; see above. Due to this finding, WARP was officially adopted within the alignment process used for the more recent Gemma-2 model \[31\]; see [here](https://huggingface.co/blog/gemma2) for more details.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F972c814e-d2a4-429d-9907-dba6cd97de4f_2150x730.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F972c814e-d2a4-429d-9907-dba6cd97de4f_2150x730.png)

(from \[48\])

> _“We periodically reset the online model to an exponentially moving average (EMA) of itself, then reset the EMA model to the initial model.”_ - from \[47\]

**What came before this?** Authors in \[10\] were not the first to try to address this balance between reward and KL. Elastic reset \[47\] eliminates KL divergence from the RLHF objective and instead periodically resets the model weights to the EMA of itself throughout the finetuning process, which enables higher rewards to be achieved with less drift. Alternatively, authors in \[48\] propose a similar strategy for updating the anchor model within RLHF by periodically setting the anchor model equal to the EMA of model weights; see above. Notably, this update strategy is quite similar to the first stage WARP.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde281fd6-a4cd-4cda-be2d-1b55a7d79451_1846x580.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde281fd6-a4cd-4cda-be2d-1b55a7d79451_1846x580.png)

(from \[51\])

In \[50\], we see that merging pre and post-RLHF models—_similarly to WiSE-FT or stage three of WARP_—is an effective method of mitigating the alignment tax, allowing us to achieve high reward without significantly degrading benchmark performance relative to the pre-RLHF model. Authors also perform some analysis to show that the benefits of model averaging are related to an increase in feature diversity, which aligns with findings in \[42\]. Going further, we see in \[51\] that LLMs finetuned via SFT also tend to suffer from an alignment tax, which can be solved via a model merging strategy that:

*   Trains many SFT “sub-models” on different subsets of instruction-following data, thus mitigating sources of bias in the dataset used for SFT.
    
*   Merges all SFT sub-models into a single model via a weighted average.
    

As we can see from this work, the ideas explored in WARP have their foundations in prior research. However, WARP combines these ideas into a novel, unified framework that is highly effective in the LLM alignment domain.

#### Advanced Topics in Model Merging Research

Beyond the core papers explored above, a variety of research has been published on the topic of model merging in the last few years, especially due to the technique’s popularity in the LLM domain. In this section, we outline several recent areas of research related to model merging that are especially interesting.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2a9721c-206a-49a8-9dd0-704318945131_1090x1430.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2a9721c-206a-49a8-9dd0-704318945131_1090x1430.png)

(from \[52, 53\])

**Merging reward models.** So far, we have seen several examples of techniques for integrating model merging into the preference tuning process for LLMs \[10, 47, 48, 50\]. A few recent papers have focused on the more specific application of merging for creating better reward models—_a single step within the RLHF process_. The two primary techniques in this space, which are quite similar to each other, are called **W**eight **A**veraged **R**eward **M**odels (WARM) \[52\] and Reward Soups \[53\]. This research, which aims to lessen the chances of reward hacking, independently trains several reward models with different hyperparameters and merges all of these models together; see above. Reward models obtained via such a merging strategy are found to be more robust and reliable compared to individual reward models, leading to improved downstream results with RLHF.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6fe13695-1db4-427a-bcae-70eebcf6da74_2228x860.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6fe13695-1db4-427a-bcae-70eebcf6da74_2228x860.png)

(from \[38, 39\])

**Merging independent models.** Usually, we merge models that are derived from the same pretrained model. These models share the same architecture and are guaranteed to be linearly mode connected. However, recent work has explored merging models that are trained independently. The foundation of this work is the idea of _permutation invariance_ \[38, 39\], which claims that all models trained to convergence via SGD—_even those that are independently trained_—lie in the same region of the loss landscape if we permute (or shuffle) their weights properly; see above. If we find the proper permutation such that this is the case (i.e., see \[38\] for an algorithm to do this), these models are linearly mode connected—_as shown in \[39\]_—and we can successfully merge their weights.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f0c8237-c3b5-4ff7-9a37-e74376fbdede_1874x782.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f0c8237-c3b5-4ff7-9a37-e74376fbdede_1874x782.png)

(from \[40\])

More recent work \[40\] has also explored distillation-based strategies for merging several LLMs with different architectures by _i)_ computing the outputs of several source LLMs for each input text and _ii)_ training a target LLM to align with the output distributions produced by each of the source models; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F607153bc-76c7-47fe-b27c-b959d7fdc7c8_1608x1112.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F607153bc-76c7-47fe-b27c-b959d7fdc7c8_1608x1112.png)

(from \[32\])

**Evolutionary model merging.** Evolutionary algorithms are a class of optimization techniques that serve as an alternative to gradient-based optimization. Instead of using gradient descent to train a model, we simply evolve a population of models by iteratively selecting (and modifying) a dynamic, high-performing—_as measured by a loss or objective function_—subset of models within the population; see [here](https://blog.evolv.ai/ai-101-intro-to-evolutionary-algorithms) for an overview of this topic. In \[32\], authors explore the intersection of evolutionary algorithms and model merging, finding that we can use these techniques to automatically discover effective combinations of open-source models; see above. In particular, we can evolve the weights used to merge a group of models at each layer, finding a more optimal model combination during the merging process.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5c476308-7fce-49c5-bf25-7287a61d1cde_1612x692.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5c476308-7fce-49c5-bf25-7287a61d1cde_1612x692.png)

(from \[34\])

**Adaptive (or automatic) merging.** Instead of evolving the coefficients used for model merging, authors in \[34\] show that we can use unsupervised methods to discover optimal merging parameters without the need for any training data. This technique, called AdaMerge, is based on the idea of [entropy minimization](https://proceedings.neurips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf). We take a bunch of unlabeled data, compute the model’s predictions on this data, and train the merging coefficients to minimize the entropy of these predictions; see above. Given that this technique is mostly applied to classification tasks with [ViT](https://huggingface.co/docs/transformers/en/model_doc/vit) models in \[34\], this means that we train the model to place most of its probability mass on a single class, as opposed to being “unsure” (i.e., assigning relatively high probability to several classes). This technique is shown to yield competitive results in the multi-task learning domain for classification models.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2879ff64-a27f-4c6a-be74-a29cd10269b6_2142x772.png)

](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2879ff64-a27f-4c6a-be74-a29cd10269b6_2142x772.png)

(from \[33\])

**Passthrough merging.** LLM practitioners have recently explored the idea of combining several LLMs into one by simply concatenating their layers. This approach yields models with weird parameter counts (e.g., [Goliath-120B](https://huggingface.co/alpindale/goliath-120b) or [Solar-10.7B](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)), referred to as Frankenstein models (or “FrankenMerges”) by the AI community; see [here](https://huggingface.co/blog/mlabonne/merge-models#4-passthrough) for details. Passthrough merging has not been explored extensively in the literature, but authors in \[33\] do discuss this technique—_referred to as depth up-scaling in their report_—for Solar-10.7B; see above. Nonetheless, passthrough merging has been used extensively by practitioners, resulting in the creation of several impressive models when paired with additional training.

Concluding Thoughts
-------------------

We have seen a lot of information within this overview, but the fundamental idea we have covered is incredibly simple—_taking a (weighted) average of weights for a group of models is an effective way of combining them_. This simple idea has spawned an entire field of research, impacting a variety of important topics like neural network training and optimization, transfer learning, sparsity and pruning, LLM alignment, and more. To summarize some of this amazing research, a few basic takeaways from model merging research over the years have been outlined below.

**A new technique?** The idea of model merging is not new whatsoever. In fact, this idea is nearly as old as the concept of an ensemble \[7\]. The intuition behind model merging is quite simple. We know that forming an output-space ensemble of several machine learning models can benefit performance, but this approach increases inference costs. Instead, we can take a weight-space ensemble via a (weighted) average of the models’ parameters, forming a single merged model with performance that rivals that of an ensemble. Many papers have extended this concept over time, but the fundamental idea and intuition remains the same!

**Why does this work?** To understand the effectiveness of model merging, we need to understand two fundamental concepts: _mode connectivity_ and _sparsity_. Linear mode connectivity tells us that different finetuned models lie in a similar basin of the loss landscape and that all models obtained via interpolation of these models have constant performance. For this reason, merging—_or interpolating_—models just yields another high-performing model! Plus, the fact that deep networks (including LLMs) exhibit high levels of sparsity implies that the likelihood of a conflict occurring between merged model parameters is relatively low.

**Where should we start?** The most basic implementation of model merging that we can derive is a uniform average of model weights, which works well in many cases. Beyond this simple approach, the fundamental idea behind model merging is best characterized by the concept of a task vector \[1\]. After creating task vectors for different finetuned models, we can perform arbitrary arithmetic and create a variety of interesting model combinations or merges, as well as perform (weighted) model averaging. To determine the optimal weights to use for model merging, we can simply measure performance on a hold-out validation set.

**Advanced strategies.** If simple model merging techniques based upon averaging or task vectors do not yield our desired result, we can explore more advanced model merging techniques, such as TIES, DARE, SLERP, and more. These strategies improve upon baseline performance by increasing sparsity, accounting for interference between merged parameters, maintaining useful geometric properties of merged model weights, and more. Going further, new merging techniques (e.g., Passthrough merging) are being proposed all the time. We should always try simple techniques first, but more advanced merging strategies have been applied successfully in practice and may be beneficial to some use cases.

#### New to the newsletter?

Hi! I’m [Cameron R. Wolfe](https://cameronrwolfe.me/), Deep Learning Ph.D. and Machine Learning Scientist at [Netflix](https://research.netflix.com/research-area/nlp-and-conversations). This is the Deep (Learning) Focus newsletter, where I help readers better understand important topics in AI research. If you like the newsletter, please subscribe, share it, or follow me on [X](https://twitter.com/cwolferesearch) and [LinkedIn](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)!

#### Bibliography

\[1\] Ilharco, Gabriel, et al. "Editing models with task arithmetic." _arXiv preprint arXiv:2212.04089_ (2022).

\[2\] Yadav, Prateek, et al. "Ties-merging: Resolving interference when merging models." _Advances in Neural Information Processing Systems_ 36 (2024).

\[3\] Yu, Le, et al. "Language models are super mario: Absorbing abilities from homologous models as a free lunch." _Forty-first International Conference on Machine Learning_. 2024.

\[4\] Zhou, Chunting, et al. "Lima: Less is more for alignment." _Advances in Neural Information Processing Systems_ 36 (2024).

\[5\] Wortsman, Mitchell, et al. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." _International conference on machine learning_. PMLR, 2022.

\[6\] Dietterich, Thomas G. "Ensemble methods in machine learning." _International workshop on multiple classifier systems_. Berlin, Heidelberg: Springer Berlin Heidelberg, 2000.

\[7\] Utans, Joachim. "Weight averaging for neural networks and local resampling schemes." _Proc. AAAI-96 Workshop on Integrating Multiple Learned Models. AAAI Press_. Citeseer, 1996.

\[8\] Wortsman, Mitchell, et al. "Robust fine-tuning of zero-shot models." _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_. 2022.

\[9\] Jang, Dong-Hwan, Sangdoo Yun, and Dongyoon Han. "Model Stock: All we need is just a few fine-tuned models." _arXiv preprint arXiv:2403.19522_ (2024).

\[10\] Ramé, Alexandre, et al. "WARP: On the Benefits of Weight Averaged Rewarded Policies." _arXiv preprint arXiv:2406.16768_ (2024).

\[11\] Garipov, Timur, et al. "Loss surfaces, mode connectivity, and fast ensembling of dnns." _Advances in neural information processing systems_ 31 (2018).

\[12\] Frankle, Jonathan, et al. "Linear mode connectivity and the lottery ticket hypothesis." _International Conference on Machine Learning_. PMLR, 2020.

\[13\] Neyshabur, Behnam, Hanie Sedghi, and Chiyuan Zhang. "What is being transferred in transfer learning?." _Advances in neural information processing systems_ 33 (2020): 512-523.

\[14\] Rofin, Mark, Nikita Balagansky, and Daniil Gavrilov. "Linear interpolation in parameter space is good enough for fine-tuned language models." _arXiv preprint arXiv:2211.12092_ (2022).

\[15\] Sun, Mingjie, et al. "Massive activations in large language models." _arXiv preprint arXiv:2402.17762_ (2024).

\[16\] Frankle, Jonathan, and Michael Carbin. "The lottery ticket hypothesis: Finding sparse, trainable neural networks." _arXiv preprint arXiv:1803.03635_ (2018).

\[17\] LeCun, Yann, John Denker, and Sara Solla. "Optimal brain damage." _Advances in neural information processing systems_ 2 (1989).

\[18\] Hassibi, Babak, David G. Stork, and Gregory J. Wolff. "Optimal brain surgeon and general network pruning." _IEEE international conference on neural networks_. IEEE, 1993.

\[19\] Frantar, Elias, and Dan Alistarh. "Sparsegpt: Massive language models can be accurately pruned in one-shot." _International Conference on Machine Learning_. PMLR, 2023.

\[20\] Sun, Mingjie, et al. "A simple and effective pruning approach for large language models." _arXiv preprint arXiv:2306.11695_ (2023).

\[21\] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In NeurIPS, 2022.

\[22\] Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." _arXiv preprint arXiv:1803.05407_ (2018).

\[23\] David Ruppert. Efficient estimations from a slowly convergent robbins-monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.

\[24\] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

\[25\] Laine, Samuli and Aila, Timo. Temporal Ensembling for Semi-Supervised Learning. arXiv:1610.02242 \[cs\], October 2016. arXiv: 1610.02242.

\[26\] Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." _Advances in neural information processing systems_ 30 (2017).

\[27\] Sanyal, Sunny, et al. "Early weight averaging meets high learning rates for llm pre-training." _arXiv preprint arXiv:2306.03241_ (2023).

\[28\] Ouyang, Long, et al. "Training language models to follow instructions with human feedback." _Advances in neural information processing systems_ 35 (2022): 27730-27744.

\[29\] Kirk, Robert, et al. "Understanding the effects of rlhf on llm generalisation and diversity." _arXiv preprint arXiv:2310.06452_ (2023).

\[30\] Team, Gemma, et al. "Gemma: Open models based on gemini research and technology." _arXiv preprint arXiv:2403.08295_ (2024).

\[31\] Team, Gemma, et al. "Gemma 2: Improving open language models at a practical size." _arXiv preprint arXiv:2408.00118_ (2024).

\[32\] Akiba, Takuya, et al. "Evolutionary optimization of model merging recipes." _arXiv preprint arXiv:2403.13187_ (2024).

\[33\] Kim, Dahyun, et al. "Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling." _arXiv preprint arXiv:2312.15166_ (2023).

\[34\] Yang, Enneng, et al. "Adamerging: Adaptive model merging for multi-task learning." _arXiv preprint arXiv:2310.02575_ (2023).

\[35\] Rame, Alexandre, et al. "Diverse weight averaging for out-of-distribution generalization." _Advances in Neural Information Processing Systems_ 35 (2022): 10821-10836.

\[36\] Ramé, Alexandre, et al. "Model ratatouille: Recycling diverse models for out-of-distribution generalization." _International Conference on Machine Learning_. PMLR, 2023.

\[37\] Ken Shoemake. Animating rotation with quaternion curves. In SIGGRAPH, 1985.

\[38\] Ainsworth, Samuel K., Jonathan Hayase, and Siddhartha Srinivasa. "Git re-basin: Merging models modulo permutation symmetries." _arXiv preprint arXiv:2209.04836_ (2022).

\[39\] Entezari, Rahim, et al. "The role of permutation invariance in linear mode connectivity of neural networks." _arXiv preprint arXiv:2110.06296_ (2021).

\[40\] Wan, Fanqi, et al. "Knowledge fusion of large language models." _arXiv preprint arXiv:2401.10491_ (2024).

\[41\] Cha, Junbum, et al. "Swad: Domain generalization by seeking flat minima." _Advances in Neural Information Processing Systems_ 34 (2021): 22405-22418.

\[42\] Lin, Yong, et al. "Spurious feature diversification improves out-of-distribution generalization." _arXiv preprint arXiv:2309.17230_ (2023).

\[43\] Zaman, Kerem, Leshem Choshen, and Shashank Srivastava. "Fuse to forget: Bias reduction and selective memorization through model fusion." _arXiv preprint arXiv:2311.07682_ (2023).

\[44\] Eeckt, Steven Vander. "Weight averaging: A simple yet effective method to overcome catastrophic forgetting in automatic speech recognition." _arXiv preprint arXiv:2210.15282_ (2022).

\[45\] Lawson, Daniel, and Ahmed H. Qureshi. "Merging decision transformers: Weight averaging for forming multi-task policies." _2024 IEEE International Conference on Robotics and Automation (ICRA)_. IEEE, 2024.

\[46\] Nikishin, Evgenii, et al. "Improving stability in deep reinforcement learning with weight averaging." _Uncertainty in artificial intelligence workshop on uncertainty in Deep learning_. 2018.

\[47\] Noukhovitch, Michael, et al. "Language model alignment with elastic reset." _Advances in Neural Information Processing Systems_ 36 (2024).

\[48\] Gorbatovski, Alexey, et al. "Learn your reference model for real good alignment." _arXiv preprint arXiv:2404.09656_ (2024).

\[49\] Xiao, Shitao, et al. "Lm-cocktail: Resilient tuning of language models via model merging." _arXiv preprint arXiv:2311.13534_ (2023).

\[50\] Lin, Yong, et al. “Mitigating the alignment tax of rlhf.” arXiv preprint _arXiv:2309.06256_ (2024).

\[51\] Fu, Tingchen, et al. "Disperse-Then-Merge: Pushing the Limits of Instruction Tuning via Alignment Tax Reduction." _arXiv preprint arXiv:2405.13432_ (2024).

\[52\] Ramé, Alexandre, et al. "Warm: On the benefits of weight averaged reward models." _arXiv preprint arXiv:2401.12187_ (2024).

\[53\] Rame, Alexandre, et al. "Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards." _Advances in Neural Information Processing Systems_ 36 (2024).

\[54\] Goddard, Charles, et al. "Arcee's MergeKit: A Toolkit for Merging Large Language Models." _arXiv preprint arXiv:2403.13257_ (2024).

[1](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-1-147448898)

[Multi-task learning](https://arxiv.org/abs/2404.18961) is an alternative technique that avoids this issue with increased inference cost. However, multi-task learning is not a clean alternative to a model ensemble, as the situations in which these two techniques would be applied can be different (e.g., we can create an ensemble of models trained on only one task).

[2](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-2-147448898)

The “deep learning era” in this context is considered to be anything after [AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) was proposed in 2012.

[3](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-3-147448898)

Here, we assume that the networks have identical size / architecture and are just trained independently with different random seeds (potentially with some differences in other hyperparameters or settings).

[4](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-4-147448898)

By this, we mean that any weights along this connected path between the weights of the original networks yield a model with training and test accuracy that match or improve upon that of the original models.

[5](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-5-147448898)

This overview was the first Deep (Learning) Focus newsletter that was ever released!

[6](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-6-147448898)

Performance is measured both on the normal ImageNet test set, as well as several other ImageNet-related test sets with various distribution shifts (e.g., [ImageNet-V2](https://imagenetv2.org/), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), and more).

[7](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-7-147448898)

Prior state-of-the-art was 90.88%, achieved by [CoAtNet](https://arxiv.org/abs/2106.04803). Today, state-of-the-art performance on ImageNet is [slightly higher](https://paperswithcode.com/sota/image-classification-on-imagenet) (around 92%).

[8](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-8-147448898)

See [this paper](https://arxiv.org/abs/2305.14201) as a great example of this phenomenon. We can outperform GPT-4 on particular target domains via finetuning!

[9](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-9-147448898)

Also, changing the coefficient used during model merging is easy! No extra training is required, whereas tweaking hyperparameters requires the model to be retrained.

[10](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-10-147448898)

Notably, all comparisons are performed in a layer-wise fashion, meaning that we only compare the weights of different finetuned models within the same layer.

[11](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-11-147448898)

This idea is explored in \[22\] for deep learning, but this technique in general is not new and can be seen in much earlier work; e.g., see \[23\] for an example.

[12](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-12-147448898)

Again, this finding has a strong connection to research on model pruning, where we see [very similar observations](https://arxiv.org/abs/1810.05270)!

[13](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-13-147448898)

This sign vector is just a vector with the same size as a task vector, where each element is either -1 or 1. The sign vector indicates the majority sign of each parameter across task vectors, as dictated by the elect sign step.

[14](https://cameronrwolfe.substack.com/p/model-merging#footnote-anchor-14-147448898)

In most cases, the size of SFT datasets are within the range of a few thousand to a few tens of thousands of examples, but we have seen SFT applied successfully with [as few as 1K examples](https://arxiv.org/abs/2305.11206) (i.e., the algorithm is very data efficient).

### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 2 years ago

I contextualize and explain important topics in AI research.

63 Likes

·

[8 Restacks](https://substack.com/note/p-147448898/restacks?utm_source=substack&utm_content=facepile-restacks)

63

[

8

](https://cameronrwolfe.substack.com/p/model-merging/comments)

8

Share
> Converted by [Clearly Reader](https://clearlyreader.com)
