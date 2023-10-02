---
title:  "Neural Decoding"
mathjax: true
layout: post
categories: media
---

![NN](https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/NNdecoding.jpg)


![VMPCA](https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/VMPCA.jpg)

---
title:  "Neural Decoding"
mathjax: true
layout: post
categories: media
---

# Setup


```python
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

# Matplotlib colors
colors = plt.cm.rainbow(np.linspace(0, 1, 8))
cmap = ListedColormap(colors)
```


```python
# Load Data
data_path = '/Users/darinmomayezi/Documents/School/Centuri Summer School/Computational Neuroscience/data/'
dataset1 = si.loadmat(data_path + 'Dataset1.mat')
header, version, globals, directions, R = dataset1['__header__'], dataset1['__version__'], dataset1['__globals__'], dataset1['direction'], dataset1['R']
neurons, trials = R.shape
directions = directions.reshape(158,)
```

# Getting Started: Visualize the Data


```python
fig = plt.figure(figsize=(9, 7))
plt.imshow(R.T)

plt.xlabel('Neruon')
plt.ylabel('Trial')
plt.title('Data')
```




    Text(0.5, 1.0, 'Data')




    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_4_1.png" width="400" height="250" />
</p>
    


The above data shows the firing rate (Hz) of each neuron during each trial. This is far too much data to analyze at once so let's try other ways to explore the data.

# Mean Firing Rate of First Neuron in Each Direction

Let's explore how the firing rate of a single neuron averages over all trials changes with the direction of the task. You can change the neuron to plot by changing the 'neuron' variable.


```python
# Directional mean firing rate
neuron = 1  # Neuron of interest
fig, ax = plt.subplots(nrows=1, ncols=1)
directional_mean_firing_rate = []
directional_std = []
trial_direction_dict = {}
for i in range(1, 9):
    directional_mean_firing_rate.append(np.mean(R[neuron][directions == i]))
    directional_std.append(np.std(R[neuron][directions == i]))

for trial, direction in enumerate(directions):
    if direction not in trial_direction_dict.keys():
        trial_direction_dict[direction] = [trial]
    else:
        trial_direction_dict[direction].append(trial)
        
ordered_trial_dict = {}
for direction in range(1, 9):
    ordered_trial_dict[direction] = trial_direction_dict[direction]
    
for index, direction in enumerate(directions):
    if direction not in trial_direction_dict.keys():
        trial_direction_dict[direction] = [index]
    else:
        trial_direction_dict[direction].append(index)
  
plt.errorbar([i for i in range(1, 9)], directional_mean_firing_rate, yerr=directional_std)
ax.plot(np.linspace(1, 8, 8), directional_mean_firing_rate, marker='o', color='blue')

plt.xlabel('Directions')
plt.ylabel('Mean Firing Rate (Hz)')
plt.title(f'Mean Firing Rate of Neuron {neuron}')
```




    Text(0.5, 1.0, 'Mean Firing Rate of Neuron 1')




<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_7_1.png" width="400" height="250" />
</p>
    


### Let's plot the rest of the neurons


```python
fig, ax = plt.subplots(nrows=15, ncols=10, figsize=(20, 30))
for neuron in range(neurons):
    directional_mean_firing_rate = []
    directional_std = []
    for i in range(1, 9):
        directional_mean_firing_rate.append(np.mean(R[neuron][directions == i]))
        directional_std.append(np.std(R[neuron][directions == i]))
    
    directional_mean_firing_rate = np.array(directional_mean_firing_rate)
    directional_std = np.array(directional_std)
    
    # ax[neuron//10][neuron%10].errorbar([i for i in range(1, 9)], directional_mean_firing_rate, yerr=directional_std)  # errorbar
    ax[neuron//10][neuron%10].plot(np.linspace(1, 8, 8), directional_mean_firing_rate, color='blue')
    ax[neuron//10][neuron%10].fill_between(range(1, 9), directional_mean_firing_rate - directional_std, directional_mean_firing_rate + directional_std)
    ax[neuron//10][neuron%10].set_xticks(np.linspace(1, 8, 8))
```



<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_9_0.png" width="800" height="450" />
</p>


One neuron has flatlined! Not a hard worker that one. These plots show that neurons have prefered directions because they fire at a greater rate in some directions than others. Therefore, we fit the data using a Von Mises distrbution, which is a continuous probability distrbution on the circle. A prewritten program will fit the data to the Von Mises distrbituion using the $$\theta_0$$ parameter in the Von Mises equation below. $$\theta_0$$ will tell us the 'preferred' direction of each neuron and we call the fits plotted in orange 'tuning curves'. 

$$VM = bs + e^{ \kappa \cos (2 ( \theta - \theta_0) / \theta_b)} (f_{ max} - bs)$$


```python
# Von Mises fit algorithm

def tuning_function(theta, theta0, kappa, fmax, bsl, theta_bound):
    # Von Mises, with kappa the concentration, theta0 the location
    # fmax the firing rate at pref ori, bsl the min firing rate (not the baseline, which was substracted) 
    tf = bsl + np.exp(kappa*(np.cos(2*np.pi/theta_bound*(theta-theta0))-1)) * (fmax-bsl)
    return tf

# we will use the tutorial from https://lmfit.github.io/lmfit-py/model.html :
from lmfit import Model, Parameters

def fit_tc(array, init_kappa, theta_bound=2*np.pi):
    """
    Fits the data points in `array` to `tuning_function`.
    
    set theta_bound to 2*np.pi for fitting signed angles (eg dirrection)
    or to np.pi for an unsigned one (like orientation)
    """
              
    theta = np.linspace(0, theta_bound, len(array), endpoint=False)
    mod = Model(tuning_function)
    pars = Parameters()
    #               name    default           vary  min   max
    pars.add_many(('theta0', theta[np.argmax(array)], True, 0., theta_bound),
                  ('kappa', init_kappa, True,  .1, 5.),
                  ('fmax', np.max(array), True, 0.0, 2*np.max(array)+5),
                  ('bsl', np.min(array), True, 0.0, np.max(array)+5),
                  ('theta_bound', theta_bound, False))
                
    out = mod.fit(array, pars, theta=theta, nan_policy='omit', max_nfev=50000)
    
    return out.best_values
```


```python
fig, ax = plt.subplots(figsize=(20, 10), nrows=15, ncols=10)
fig1, ax1 = plt.subplots(15, 10, figsize=(23, 23), squeeze = False, subplot_kw={'projection': 'polar'})
theta_0_array = []
for neuron in range(neurons):
    
    directional_mean_firing_rate = []
    for i in range(1, 9):
        directional_mean_firing_rate.append(np.mean(R[neuron][directions == i]))
    
    fr_dict = {}  # firing rates for each direction
    
    for directional in range(1, 9):
        fr_dict[directional] = R[neuron][directions == directional]
    
    fr_array = np.zeros(len(fr_dict.keys()))      
    for direction, fr_list in fr_dict.items():
        fr_array[direction-1] = np.mean(fr_list)
        
    N_directions = 8
    theta_bound = 2*np.pi
    theta = np.linspace(0, theta_bound, N_directions, endpoint=False)
    array = tuning_function(theta, theta0=np.pi/4, kappa=2., fmax=20., bsl=1., theta_bound=theta_bound) + 0.1*np.abs(np.random.randn(N_directions))
                    
    fitted_params = fit_tc(fr_array, 10)
    theta_0_array.append(fitted_params['theta0'])
    ax[neuron//10][neuron%10].plot(theta, fr_array)
    theta_more = np.linspace(0, theta_bound, 60, endpoint=True)
    ax[neuron//10][neuron%10].plot(theta_more, tuning_function(theta_more, **fitted_params))

    ax[neuron//10][neuron%10].set_yticks([0])
    ax[neuron//10][neuron%10].set_xticks([0])
    
    # Polar plots
    ax1[neuron//10][neuron%10].scatter(theta, np.array(directional_mean_firing_rate), marker='o')
    ax1[neuron//10][neuron%10].axvline(fitted_params['theta0'], color = 'red')
    ax1[neuron//10][neuron%10].plot(theta_more, tuning_function(theta_more, **fitted_params), color='orange')
fig.supxlabel('Direction')
fig.supylabel('Mean firing rate')
fig.suptitle('Tuning Functions: Von Mises Method')
plt.show()
```


    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_12_0.png" width="800" height="450" />
</p>
    



    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_12_1.png" width="800" height="450" />
</p>
    


The first set of plots shows the true data vs. the fitted data, which shows that the Von Mises distribution provides a good fit. Therefore, the $$\theta_0$$ variable does show the preferred direction. The second group of plots is polar plots that show the Von Mises fit in its circular geometry. The red line shows the preferred direction based on the $$f_{\text{max}}$$ parameter and provides an intuitive interpretation of the neuron firing rate. Remember each plot is a different neuron, so we are finding the preferred direction of each neuron. We can make an even more concise graph by plotting the distribution of preferred directions on the circle. 


```python
plt.hist(theta_0_array, bins=20)
plt.title('$\\theta$ distribution', fontweight = 'bold')
plt.show()
```


    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_14_0.png" width="400" height="250" />
</p>
    


Now we wrap this $$\theta$$ distribution around the circle. 


```python
radii = [10.,  5.,  3.,  2.,  6.,  9., 11., 11., 15.,  8.,  5.,  8.,  3.,
         6.,  1.,  6.,  4., 14.,  8.,  8.]

angles = [0,0.31415927, 0.62831853, 0.9424778 , 1.25663706,
        1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
        3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822971,
        4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,]

width = (2*np.pi) / len(radii)

ax = plt.subplot(111, polar=True)
bars = ax.bar(angles, radii, width = width)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_alpha(0.8)
ax.set_title('$\\theta$ distribution', fontweight = 'bold')

plt.show()
```


    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_16_0.png" width="400" height="250" />
</p>
    


Directions near $$135 \degree$$ and $$115 \degree$$ are clearly overrepresented, which means many more neurons are required to perform these movements. This may be due to the fact that diagonal movements require a movement of both joints in the arm, namely the elbow and shoulder. Therefore, we can explain this through the geometry of the arm, but we didn't assume that before! It came directly from the neuronal activity, however, it is only a conjecture until we perform an experiment to validate this hypothesis. 

At this point we have successfully uncovered the circular geometry of this activity solely from neuronal activity by: 

1. Exploring the data in a smart way (activity vs. direction, averaged over trials) and noticing a directionality
2. Using the Von Mises distribution (motivated by step 1) to find a parameter $\theta_0$ (preferred direction of neuron)
3. Plotting useful information in the appropriate geometrical format (polar plot for the circular geometry of the Von Mises distribution) to obtain interpretible information. 
 
### Now, can we decode the detailed movement of the monkey's arm solely from neuronal activity?

In the following section we will show this is possible by using two methods:
1. Logistic Regression
2. Neural Networks

This task will inevitably require some form of dimensionality reduction since we hope to obtain one piece of information as output (the direction that the moneky points) from 143 neurons of input. So, we first employ principle component analysis to get a naive estimate of how high dimensional the input is. We say it is 'naive' because there is no reason that a linear dimensionality reduction method will represent the neuronal data well, i.e. we expect the neuronal data to be moderately to highly non-linear. 

# Principle Components Analysis (PCA)


```python
R_prime = normalize(R.T)
R_prime = R_prime - np.mean(R_prime, axis=0)
pca = PCA(n_components=2, svd_solver='full')
pca.fit(R_prime)  # fit normalized data to princ_comps
explained = pca.explained_variance_ratio_  # variance explained by each component
princ_comps = pca.components_
projected = pca.transform(R_prime)
print(projected.shape)
fig = plt.subplot()
print(sum(explained))

for trial_index, trial in enumerate(projected):
    # do coloring
    for direction, trials in trial_direction_dict.items():
        for trial_sample in trials:
            if trial_sample == trial_index:
                fig.scatter(projected[trial_index, 0], projected[trial_index, 1], color=cmap(direction))


# fig4.scatter(projected[:, 0], projected[:, 1], marker='o')
fig.set_title(f'Explained Variance {round(sum(explained), 2)}')
fig.legend()
```

    (158, 2)
    0.43838951138059024


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.





    <matplotlib.legend.Legend at 0x134deea70>




    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_19_3.png" width="400" height="250" />
</p>
    


We are correct in our assumption that PCA will not represent the data very well. The first two components explain 44% of the variance in the data, which isn't bad, but playing around with the code shows that we need about 40 components to explain 90% or more of the data. PCA is still useful, however. One thing to notice is that information is fairly well spread over the neurons precisely because we need 40+ components to explain 90% of the data. Why is that? Well, as a phycisist who doesn't know about neuroscience I asked the question, 'how can we possibly insert electrodes in the brain and happen upon THE 143 neurons that produce the monkey's movement?' The answer is that information is actually spread out pretty well over parts of the brain. This means that the activity of most neurons in the motor cortex are somehow correlated to each movement we do. Perhaps this redundancy is an evolutionary advantage since, while less efficient, redundancy provides more security is some wiring goes awry. Anyways, this is something interesting to notice. Another reason PCA is useful is because now we can visualize the data in a two-dimensional space. Since we can't visually percieve more than three dimensions this is very useful. 

### Remember, we have a continuous represenation of the data with $$\theta_0$$.

Let's recall that the original task is performed with discrete directions 1-8 around the circle, but with the Von Mises fit we can create a continuous representation of the data through interpolation. Therefore, with the same 143 neurons we can predict any direction the monkey could move if it were allowed to move in any direction around the circle. 


```python
# Interpolate data with Von Mises fit
scaler = StandardScaler()

VM = []
init_kappa = 3.5
N_directions = 8
theta_bound = 2 * np.pi
theta = np.linspace(0, theta_bound, N_directions, endpoint = False)
theta_more = np.linspace(0, theta_bound, 60, endpoint = True)


for neuron in range(neurons):
        
    ########################
    # Retrieve mean firing rates for each neuron at each direction
    mean_firing = []
    
    for i in range(1, 9): # Moving along 'directions'
        mean_firing.append(np.mean(R[neuron][directions == i]))
    mean_firing = np.array(mean_firing)

    fitted_params = fit_tc(mean_firing, init_kappa = init_kappa)  # VM fit
    VM.append(tuning_function(theta_more, **fitted_params))
    
VM = np.array(VM)

VM_projected = scaler.fit_transform(VM.T)
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(VM_projected)

# Transform the scaled data to the principal components
data_t_reduced_VM = pca.transform(VM_projected)

fig = plt.figure(figsize = (12,3))
fig.subplots_adjust(wspace = 0.8)

plt.subplot(121)
colors_m = plt.cm.rainbow(np.linspace(0, 1, len(theta_more)))
plt.scatter(data_t_reduced_VM[:, 0], data_t_reduced_VM[:, 1], color = colors_m, s = 1e2)


plt.title('PCA - VM', fontweight = 'bold')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Add legends as style (this can be ommited)
handles = []
for i in range(8):
    exec('patch{} = mpatches.Patch(color = colors[{}], label =  "{}")'.format(i + 1, i, i + 1))
    handles.append(eval('patch{}'.format(i + 1)))

plt.legend(handles = handles, loc = 'center left', bbox_to_anchor = (1.05, 0.5), frameon = False)

plt.subplot(122)
plt.bar([i + 1 for i in range(n_components)], pca.explained_variance_ratio_)
plt.title('Explained Variance ({})'.format(round(pca.explained_variance_ratio_.sum(), 3)), fontweight = 'bold')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.show()
```


    
<p align="center">
  <img src="https://raw.githubusercontent.com/darin-momayezi/darin-momayezi.github.io/master/images/neural_decoding_21_0.png" width="600" height="350" />
</p>
    


This ring shows us that with the neuronal data from the 143 neurons we can predict any directions in $360 \degree$ that the monkey will move! Therefore, we have created a CONTINUOUS REPRESENTATION. Let's take a minute to appreciate how we did this. First, we already have the PCA fit that we did previously on the discrete directions. Then, we applied the Von Mises fit to each neuron, taking into account its mean activity in each direction, to obtain the tuning function, i.e. the continous representation. And to visualize in the PCA space we projected this fitted data onto the first two principle components to obtain this ring. Using the principle components is one way, but not the best way, to decode the direction of motion from the neuronal activity.

# Neural Network

Now, we introduce a neural network for decoding. We have decided to formulate the network as a regression task rather than a classification task, however, both achieve the same results. The architecutre is very simple: four linear layers with three Sigmoid functions in between. IMPORTANT: the network does not end with a Sigmoid since its output is limited from 0-1 while we are predicting output from 1-8 (for each of the directions).
