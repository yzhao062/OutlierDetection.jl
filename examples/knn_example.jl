include("../src/utilities.jl")
using NearestNeighbors
using ROCAnalysis

n_samples = 100
n_features = 2
X, y = generate_data(n_samples, n_features)

# Create a KD tree for this purpose
kdtree = KDTree(X; leafsize = 15)
idxs, dists = knn(kdtree, X, k, true)

# Calculate avg distance for each instance
decision_scores = zeros(size(X)[2])
for i = 1:n_samples
    # println(i, " ", mean(dists[i]))
    decision_scores[i] = mean(dists[i])
end

# Evaluate the performance of this simple KNN detector
println("Evaluate basic KNN detection...")
roc_full = roc(convert(Array{Float64, 1}, y), decision_scores)
print("ROC score: ", auc(roc_full))
