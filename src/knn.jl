include("utilities.jl")
using NearestNeighbors

struct knn_model
    kd_tree::KDTree
    decision_scores::Float64
end

function KNN_fit(X, n_neighbors=10)

    # Create a KD tree for this purpose
    kdtree = KDTree(X; leafsize = 15)
    idxs, dists = knn(kdtree, X, k, true)

    # Calculate avg distance for each instance
    avg_dist = zeros(size(X)[2])
    for i = 1:size(X)[2]
        # println(i, " ", mean(dists[i]))
        avg_dist[i] = mean(dists[i])
    end
    avg_dist
end
