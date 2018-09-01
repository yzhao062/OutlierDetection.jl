# OutlierDetection.jl
A Julia Library for Outlier Detection (Anomaly Detection), abbreviated as JLOD.
----------------------------
### Motivation

While developing [PyOD (Python Outlier Detection Toolbox)](https://github.com/yzhao062/Pyod), I have been upset occasionally due to the complexity of writing highly efficient code in Python. Although Python is famous for its comprehensiveness and short learning curve, it also receives critics regarding its efficiency -- a lot of modules are therefore written in other languages and ported to Python, e.g., C.

This is the motivation of developing a comprehensive Outlier Detection toolbox in **Julia**, the next generation scientific computing language, famous for flexible syntax and efficiency, on par with C. See more about **[Julia](https://docs.julialang.org/en/v0.6.2/manual/introduction/#man-introduction-1)** here.

Similar to PyOD, OutlierDetection.jl also strives for: 
- **Unified and consistent APIs** across various anomaly detection algorithms for easy use.
- **Advanced functions**, e.g., **Outlier Ensemble Frameworks** to combine multiple detectors.
- **Detailed API Reference, Interactive Examples in Jupyter Notebooks** for better reliability.

Different from PyOD, we have a high hope to **achieve faster and more scalable outlier mining with JLOD**.

------------------------
### A Taste of Julia Outlier Detection

To show the syntax of Julia, I create a quick demo for using KNN detector in mining outliers. Full code could be found [here](https://github.com/yzhao062/OutlierDetection.jl/blob/master/examples/knn_example.jl)

1. Generate some artificial data and import necessary packages
    ```julia
    include("../src/utilities.jl")
    using NearestNeighbors
    using ROCAnalysis

    # create 100 samples with 2 dimensions, in which 10% are outliers
    n_samples = 100
    n_features = 2
    contamination = 0.1
    X, y = generate_data(n_samples, n_features)
    ```

2. Build a kd tree and do 10-NN detection
    ```julia
    k = 10
    # Create a KD tree for this purpose
    kdtree = KDTree(X; leafsize = 15)
    idxs, dists = knn(kdtree, X, k, true)

    # Calculate avg distance for each instance
    decision_scores = zeros(size(X)[2])
    for i = 1:n_samples
        # println(i, " ", mean(dists[i]))
        decision_scores[i] = mean(dists[i])
    end
    ```
3. Evaluate the detection result by ROC
    ```julia
    # Evaluate the performance of this simple KNN detector
    println("Evaluate basic KNN detection...")
    roc_full = roc(convert(Array{Float64, 1}, y), decision_scores)
    print("ROC score: ", auc(roc_full))
    ```
4. Evaluation result:
   ```
   Evaluate basic KNN detection...
   ROC score: 0.911
   ```
------------------------

### Development RoadMap of JLOD

It is noted that **Julia is still NOT in its maturity as of Sep 2018**, with the chance of introducing significant changes. For this reason, two development plans are running in parallel:

- **Implementing native Julia models** for some detection models
- **Porting Python models from PyOD** using PyCall as a temporary solution or as a placeholder for the future refactoring. A similar approach has been taken for [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl). 

As usual, you are more than welcome to share your ideas by opening an issue or dropping me an email at yuezhao@cs.toronto.edu :)
