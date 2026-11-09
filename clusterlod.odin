/**
 * clusterlod - a small "library" built on top of meshoptimizer to generate cluster LOD hierarchies.
 * This is a native Odin port of the C++ reference implementation (clusterlod.h) from meshoptimizer.
 *
 * Copyright (C) 2016-2026, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
 * This code is distributed under the MIT License. See notice at the end of this file.
 */
package meshopt

import "core:c"
import "core:math"

// Configuration of each cluster; maps to meshopt_buildMeshlets* parameters.
clodConfig :: struct {
	max_vertices:  int,
	min_triangles: int,
	max_triangles: int,

	// partitioning setup; maps to meshopt_partitionClusters parameters (plus optional partition sorting)
	// note: partition size is the target size, not maximum; actual partitions may be up to 1/3 larger (e.g. target 24 results in maximum 32)
	partition_spatial: bool,
	partition_sort:    bool,
	partition_size:    int,

	// clusterization setup; maps to meshopt_buildMeshletsSpatial / meshopt_buildMeshletsFlex
	cluster_spatial:       bool,
	cluster_fill_weight:   f32,
	cluster_split_factor:  f32,

	// every level aims to reduce the number of triangles by ratio, and considers clusters that don't reach the threshold stuck
	simplify_ratio:     f32,
	simplify_threshold: f32,

	// to compute the error of simplified clusters, we use the formula that combines previous accumulated error as follows:
	// max(previous_error * simplify_error_merge_previous, current_error) + current_error * simplify_error_merge_additive
	simplify_error_merge_previous: f32,
	simplify_error_merge_additive: f32,

	// amplify the error of clusters that go through sloppy simplification to account for appearance degradation
	simplify_error_factor_sloppy: f32,

	// limit error by edge length, aiming to remove subpixel triangles even if the attribute error is high
	simplify_error_edge_limit: f32,

	// clamp attribute error to match position error scale and avoid overly conservative LOD selection
	simplify_error_clamped: bool,

	// use permissive simplification instead of regular simplification (make sure to use attribute_protect_mask if this is set!)
	simplify_permissive: bool,

	// use permissive or sloppy simplification but only if regular simplification gets stuck
	simplify_fallback_permissive: bool,
	simplify_fallback_sloppy:     bool,

	// use regularization during simplification to make triangle density more uniform, at some cost to overall triangle count; recommended for deformable objects
	simplify_regularize: bool,

	// try to preserve fold lines between opposite-facing triangles during simplification, at a small performance cost
	simplify_preserve_folds: bool,

	// dilate open borders for each cluster to try to mitigate area loss; most useful for foliage, should not be enabled everywhere
	// note: this will mutate vertex_positions in place; the caller should copy positions for each cluster in the output callback as they will be overwritten by simplified clusters
	simplify_dilate_borders: bool,

	// should clodCluster.bounds be computed based on the geometry of each cluster
	optimize_bounds: bool,

	// should clodCluster.indices be optimized for locality; helps with rasterization performance and ray tracing performance in fast-build modes
	optimize_clusters:       bool,
	optimize_clusters_level: int,
}

clodMesh :: struct {
	// input triangle indices
	indices: []u32,

	// total vertex count
	vertex_count: int,

	// input vertex positions; must be 3 floats per vertex
	vertex_positions:        ^f32,
	vertex_positions_stride: int,

	// input vertex attributes; used for attribute-aware simplification and permissive simplification
	vertex_attributes:        ^f32,
	vertex_attributes_stride: int,

	// input vertex locks; allows to preserve additional seams (when not using attribute_protect_mask) or lock vertices via meshopt_SimplifyVertex_* flags
	vertex_lock: []u8,

	// attribute weights for attribute-aware simplification; maps to meshopt_simplifyWithAttributes parameters
	attribute_weights: []f32,
	attribute_count:   int,

	// attribute mask to flag attribute discontinuities for permissive simplification; mask (1<<K) corresponds to attribute K
	attribute_protect_mask: u32,
}

// To compute approximate (perspective) projection error of a cluster in screen space (0..1; multiply by screen height to get pixels):
// - camera_proj is projection[1][1], or cot(fovy/2); camera_znear is *positive* near plane distance
// - for simplicity, we ignore perspective distortion and use rotationally invariant projection size estimation
// - return: bounds.error / max(distance(bounds.center, camera_position) - bounds.radius, camera_znear) * (camera_proj * 0.5f)
clodBounds :: struct {
	// sphere bounds, in mesh coordinate space
	center: [3]f32,
	radius: f32,

	// combined simplification error, in mesh coordinate space
	error: f32,
}

clodCluster :: struct {
	// index of more refined group (with more triangles) that produced this cluster during simplification, or -1 for original geometry
	refined: int,

	// cluster bounds; should only be used for culling, as bounds.error is not monotonic across DAG
	bounds: clodBounds,

	// cluster indices; refer to the original mesh vertex buffer
	indices: []u32,

	// cluster vertex count; indices[] has vertex_count unique entries
	vertex_count: int,
}

clodGroup :: struct {
	// DAG level the group was generated at
	depth: int,

	// simplified group bounds (reflects error for clusters with clodCluster.refined == group id; error is FLT_MAX for terminal groups)
	// cluster should be rendered if:
	// 1. clodGroup.simplified for the group it's in is over error threshold
	// 2. cluster.refined is -1 *or* clodGroup.simplified for groups[cluster.refined].simplified is at or under error threshold
	simplified: clodBounds,
}

// gets called for each group in sequence
// returned value gets saved for clusters emitted from this group (clodCluster.refined)
clodOutput :: proc(output_context: rawptr, group: clodGroup, clusters: []clodCluster) -> int

// when using BVH to determine cluster visibility, traverse the tree into nodes while bounds is over error threshold; for leaves,
// group clusters should be rendered if cluster.refined is -1 *or* clodGroup.simplified for groups[cluster.refined].simplified is at or under error threshold
clodNode :: struct {
	// BVH node bounds, reflecting worst case error for all groups in the subtree
	bounds: clodBounds,

	// leaf node: index of the group this node represents (or -1 for internal nodes)
	group: int,

	// internal node: children are at nodes[child_offset..child_offset+child_count)
	child_offset: int,
	child_count:  int,
}

// default configuration optimized for rasterization / raytracing
clodDefaultConfig :: proc(max_triangles: int) -> clodConfig {
	assert(max_triangles >= 4 && max_triangles <= 256)

	config := clodConfig{}
	config.max_vertices = max_triangles
	config.min_triangles = max_triangles / 3
	config.max_triangles = max_triangles

	config.partition_spatial = true
	config.partition_size = 16

	config.cluster_spatial = false
	config.cluster_split_factor = 2.0

	config.optimize_clusters = true
	config.optimize_clusters_level = 1

	config.simplify_ratio = 0.5
	config.simplify_threshold = 0.85
	config.simplify_error_merge_previous = 1.0
	config.simplify_error_factor_sloppy = 2.0
	config.simplify_error_clamped = true
	config.simplify_permissive = true
	config.simplify_fallback_permissive = false // note: by default we run in permissive mode, but it's also possible to disable that and use it only as a fallback
	config.simplify_fallback_sloppy = true

	return config
}

clodDefaultConfigRT :: proc(max_triangles: int) -> clodConfig {
	config := clodDefaultConfig(max_triangles)

	// for ray tracing, we may want smaller clusters when that improves BVH quality further; for maximum ray tracing performance this could be reduced even further
	config.min_triangles = max_triangles / 4

	// by default, we use larger max_vertices for RT; the vertex count is not important for ray tracing performance, and this helps improve cluster utilization
	config.max_vertices = min(256, max_triangles * 2)

	config.cluster_spatial = true
	config.cluster_fill_weight = 0.5

	return config
}

// build cluster LOD hierarchy, calling output callbacks as new clusters and groups are generated
clodBuild :: proc(config: clodConfig, mesh: clodMesh, output_context: rawptr, output_callback: clodOutput) {
	assert(mesh.vertex_attributes_stride % size_of(f32) == 0)
	assert(mesh.attribute_count * size_of(f32) <= mesh.vertex_attributes_stride)
	assert(mesh.attribute_protect_mask < (u32(1) << u32(mesh.vertex_attributes_stride / size_of(f32))))

	locks := make([dynamic]u8, mesh.vertex_count)
	defer delete(locks)

	// for cluster connectivity, we need a position-only remap that maps vertices with the same position to the same index
	remap := make([dynamic]u32, mesh.vertex_count)
	defer delete(remap)
	meshopt_generatePositionRemap(&remap[0], mesh.vertex_positions, c.size_t(mesh.vertex_count), c.size_t(mesh.vertex_positions_stride))

	// set up protect bits on UV seams for permissive mode
	if mesh.attribute_protect_mask != 0 {
		max_attributes := mesh.vertex_attributes_stride / size_of(f32)
		attributes := cast([^]f32) mesh.vertex_attributes

		for i in 0 ..< mesh.vertex_count {
			r := int(remap[i]) // canonical vertex with the same position

			for j in 0 ..< max_attributes {
				if r != i && (mesh.attribute_protect_mask & (u32(1) << u32(j))) != 0 && attributes[i * max_attributes + j] != attributes[r * max_attributes + j] {
					locks[i] |= meshopt_SimplifyVertex_Protect
				}
			}
		}
	}

	dilate_offsets: [dynamic]f32
	defer delete(dilate_offsets)
	if config.simplify_dilate_borders {
		resize(&dilate_offsets, mesh.vertex_count * 4)
	}
	dilate_table: [dynamic]u64
	defer delete(dilate_table)

	// initial clusterization splits the original mesh
	cluster_indices := make([dynamic]u32)
	defer delete(cluster_indices)
	clusters := make([dynamic]Cluster)
	defer delete(clusters)
	clusterize(&clusters, &cluster_indices, config, mesh, mesh.indices)

	// compute initial precise bounds; subsequent bounds will be using group-merged bounds
	for &cluster in clusters {
		cluster.bounds = bounds_compute(mesh, cluster_indices[cluster.index_offset:cluster.index_offset + cluster.index_count], 0)
	}

	pending := make([dynamic]Cluster, 0, len(clusters))
	defer delete(pending)
	pending_indices := make([dynamic]u32, 0, int(f32(len(cluster_indices)) * config.simplify_threshold))
	defer delete(pending_indices)

	merged := make([dynamic]u32, 0, (config.partition_size + config.partition_size / 3) * config.max_triangles * 3)
	defer delete(merged)
	simplified := make([dynamic]u32)
	defer delete(simplified)

	depth := 0

	// merge and simplify clusters until we can't merge anymore
	for len(clusters) > 1 {
		group_offsets := partition(&clusters, config, mesh, cluster_indices[:], remap[:])

		clear(&pending)
		clear(&pending_indices)

		// mark boundaries between groups with a lock bit to avoid gaps in simplified result
		lock_boundary(locks[:], clusters[:], group_offsets[:], cluster_indices[:], remap[:], mesh.vertex_lock)

		// every group needs to be simplified now
		for i := 0; i + 1 < len(group_offsets); i += 1 {
			group := clusters[group_offsets[i]:group_offsets[i + 1]]

			clear(&merged)
			for j in 0 ..< len(group) {
				off := group[j].index_offset
				cnt := group[j].index_count
				append(&merged, ..cluster_indices[off:off + cnt])
			}

			target_size := int(f32(len(merged) / 3) * config.simplify_ratio) * 3

			// enforce bounds and error monotonicity
			// note: it is incorrect to use the precise bounds of the merged or simplified mesh, because this may violate monotonicity
			bounds := bounds_merge(&group[0].bounds, len(group), size_of(Cluster))

			error: f32 = 0
			simplify(&simplified, config, mesh, merged[:], locks[:], target_size, &error)
			if len(simplified) > int(f32(len(merged)) * config.simplify_threshold) {
				bounds.error = math.F32_MAX // terminal group, won't simplify further
				output_group(config, mesh, group, cluster_indices[:], bounds, depth, output_context, output_callback)
				continue // simplification is stuck; abandon the merge
			}

			// enforce error monotonicity (with an optional hierarchical factor to separate transitions more)
			bounds.error = max(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive

			// output the new group with all clusters; the resulting id will be recorded in new clusters as clodCluster.refined
			refined := output_group(config, mesh, group, cluster_indices[:], bounds, depth, output_context, output_callback)

			// now that we've output the group with the original clusters, we need to dilate simplified clusters if requested
			if config.simplify_dilate_borders {
				dilate_borders(mesh, merged[:], simplified[:], locks[:], remap[:], &bounds, dilate_offsets[:], &dilate_table)
			}

			// enqueue new clusters for further processing
			cluster_offset := len(pending)
			clusterize(&pending, &pending_indices, config, mesh, simplified[:], refined)

			// update cluster group bounds to the group-merged bounds; this ensures that we compute the group bounds for whatever group this cluster will be part of conservatively
			for j := cluster_offset; j < len(pending); j += 1 {
				pending[j].bounds = bounds
			}
		}

		delete(group_offsets)

		clusters, pending = pending, clusters
		cluster_indices, pending_indices = pending_indices, cluster_indices
		depth += 1
	}

	if len(clusters) > 0 {
		assert(len(clusters) == 1)
		cluster := clusters[0]

		bounds := cluster.bounds
		bounds.error = math.F32_MAX // terminal group, won't simplify further

		output_group(config, mesh, clusters[:], cluster_indices[:], bounds, depth, output_context, output_callback)
	}
}

// extract meshlet-local indices from cluster indices produced by clodBuild
// fills triangles[] and vertices[] such that vertices[triangles[i]] == indices[i]
// returns number of unique vertices (which will be equal to clodCluster.vertex_count)
clodLocalIndices :: proc(vertices: []u32, triangles: []u8, indices: []u32) -> int {
	return int(meshopt_extractMeshletIndices(&vertices[0], &triangles[0], &indices[0], c.size_t(len(indices))))
}

// upper bound on the number of nodes clodBuildHierarchy can produce
// level_count is the number of levels in the group DAG (max(clodGroup.depth) + 1)
clodBuildHierarchyBound :: proc(group_count: int, node_width: int, level_count: int) -> int {
	// count nodes for each tree depth; we pad by level_count at each iteration to account for unknown level distribution in the forest
	total := level_count
	for frontier := group_count; frontier > 1; frontier = (frontier + node_width - 1) / node_width {
		total += frontier + level_count
	}
	return total
}

// build a spatial hierarchy over the groups produced by clodBuild for accelerating cluster visibility
// level_count is the number of levels in the group DAG (max(clodGroup.depth) + 1)
// the result is a forest with one tree per DAG level:
// - nodes 0..level_count-1 are the roots of per-level trees, each covering groups with clodGroup.depth == level
// - remaining nodes are the tree nodes; each internal node has up to node_width children, and each leaf node refers to a single group
clodBuildHierarchy :: proc(nodes: []clodNode, groups: []clodGroup, node_width: int, level_count: int) -> int {
	// reserve space for per-level roots
	offset := level_count

	row := make([dynamic]clodNode, 0, len(groups))
	defer delete(row)
	order := make([dynamic]u32, len(groups))
	defer delete(order)

	for level in 0 ..< level_count {
		// start each tree hierarchy from the groups of that level as leaves
		clear(&row)
		for group, i in groups {
			if group.depth == level {
				node := clodNode{}
				node.bounds = group.simplified
				node.group = i
				append(&row, node)
			}
		}

		// build the tree going up one level at a time, using spatial grouping of centers as a proxy for locality
		for len(row) > 1 {
			count := len(row)
			meshopt_spatialClusterPoints(&order[0], &row[0].bounds.center[0], c.size_t(count), c.size_t(size_of(clodNode)), c.size_t(node_width))

			for i in 0 ..< count {
				nodes[offset + i] = row[int(order[i])]
			}

			clear(&row)
			for i := 0; i < count; i += node_width {
				// spatialClusterPoints guarantees that each cluster except for last one is full
				children := min(node_width, count - i)
				append(&row, merge_nodes(nodes, offset + i, children))
			}

			offset += count
		}

		// the root of the current level goes into the fixed section at the beginning
		assert(len(row) == 1)
		nodes[level] = row[0]
	}

	assert(offset <= clodBuildHierarchyBound(len(groups), node_width, level_count))
	return offset
}

// internal Cluster mirrors the processing state; indices are stored flat in cluster_indices
Cluster :: struct {
	vertices: int,

	// offset into indices vector in the processing loop
	index_offset: int,
	index_count:  int,

	refined: int,

	bounds: clodBounds,
}

vertex_pos :: proc(mesh: clodMesh, index: int) -> [3]f32 {
	positions := cast([^]f32) mesh.vertex_positions
	s := mesh.vertex_positions_stride / size_of(f32)
	return [3]f32{positions[index * s + 0], positions[index * s + 1], positions[index * s + 2]}
}

bounds_compute :: proc(mesh: clodMesh, indices: []u32, error: f32) -> clodBounds {
	bounds := meshopt_computeClusterBounds(&indices[0], c.size_t(len(indices)), mesh.vertex_positions, c.size_t(mesh.vertex_count), c.size_t(mesh.vertex_positions_stride))
	return clodBounds{center = bounds.center, radius = bounds.radius, error = error}
}

bounds_merge :: proc(bounds: ^clodBounds, count: int, stride: int) -> clodBounds {
	centers := cast(^f32) bounds
	radii := cast(^f32) (uintptr(bounds) + offset_of(clodBounds, radius))
	merged := meshopt_computeSphereBounds(centers, c.size_t(count), c.size_t(stride), radii, c.size_t(stride))

	// merged bounds error must be conservative wrt cluster errors
	error: f32 = 0
	p := bounds
	for j in 0 ..< count {
		b := cast(^clodBounds) (uintptr(p) + uintptr(j * stride))
		error = max(error, b.error)
	}

	return clodBounds{center = merged.center, radius = merged.radius, error = error}
}

clusterize :: proc(clusters: ^[dynamic]Cluster, cluster_indices: ^[dynamic]u32, config: clodConfig, mesh: clodMesh, indices: []u32, refined: int = -1) {
	index_count := len(indices)
	max_meshlets := int(meshopt_buildMeshletsBound(c.size_t(index_count), c.size_t(config.max_vertices), c.size_t(config.min_triangles)))

	meshlets := make([]meshopt_Meshlet, max_meshlets)
	defer delete(meshlets)
	meshlet_vertices := make([]u32, index_count)
	defer delete(meshlet_vertices)
	meshlet_triangles := make([]u8, index_count)
	defer delete(meshlet_triangles)

	meshlet_count: int
	if config.cluster_spatial {
		meshlet_count = int(meshopt_buildMeshletsSpatial(
			&meshlets[0], &meshlet_vertices[0], &meshlet_triangles[0],
			&indices[0], c.size_t(index_count),
			mesh.vertex_positions, c.size_t(mesh.vertex_count), c.size_t(mesh.vertex_positions_stride),
			c.size_t(config.max_vertices), c.size_t(config.min_triangles), c.size_t(config.max_triangles), config.cluster_fill_weight))
	} else {
		meshlet_count = int(meshopt_buildMeshletsFlex(
			&meshlets[0], &meshlet_vertices[0], &meshlet_triangles[0],
			&indices[0], c.size_t(index_count),
			mesh.vertex_positions, c.size_t(mesh.vertex_count), c.size_t(mesh.vertex_positions_stride),
			c.size_t(config.max_vertices), c.size_t(config.min_triangles), c.size_t(config.max_triangles), 0, config.cluster_split_factor))
	}

	cluster_offset := len(clusters)
	resize(clusters, cluster_offset + meshlet_count)

	index_offset := len(cluster_indices)
	resize(cluster_indices, index_offset + index_count)

	for i in 0 ..< meshlet_count {
		meshlet := meshlets[i]
		cluster := &clusters[cluster_offset + i]

		voff := int(meshlet.vertex_offset)
		vcount := int(meshlet.vertex_count)
		toff := int(meshlet.triangle_offset)
		tcount := int(meshlet.triangle_count)

		if config.optimize_clusters {
			meshopt_optimizeMeshletLevel(
				&meshlet_vertices[voff], c.size_t(vcount),
				&meshlet_triangles[toff], c.size_t(tcount),
				i32(config.optimize_clusters_level))
		}

		cluster.vertices = vcount

		// note: we discard meshlet-local indices; they can be recovered by the caller using clodLocalIndices
		cluster.index_offset = index_offset
		cluster.index_count = tcount * 3
		for j in 0 ..< tcount * 3 {
			cluster_indices[index_offset + j] = meshlet_vertices[voff + int(meshlet_triangles[toff + j])]
		}
		index_offset += tcount * 3

		cluster.refined = refined
	}
}

partition :: proc(clusters: ^[dynamic]Cluster, config: clodConfig, mesh: clodMesh, indices: []u32, remap: []u32) -> [dynamic]int {
	if len(clusters) <= config.partition_size {
		result := make([dynamic]int, 2)
		result[0] = 0
		result[1] = len(clusters)
		return result
	}

	// copy remapped cluster index data into a flat array for partitioning
	cluster_counts := make([]u32, len(clusters))
	defer delete(cluster_counts)
	cluster_indices := make([dynamic]u32, 0, len(indices))
	defer delete(cluster_indices)

	for cluster, i in clusters {
		cluster_counts[i] = u32(cluster.index_count)

		for j in 0 ..< cluster.index_count {
			append(&cluster_indices, remap[int(indices[cluster.index_offset + j])])
		}
	}

	// partition clusters into groups; the output is a partition id per cluster
	cluster_part := make([]u32, len(clusters))
	defer delete(cluster_part)

	positions: ^f32 = nil
	positions_count := 0
	if config.partition_spatial {
		positions = mesh.vertex_positions
		positions_count = len(remap)
	}

	partition_count := int(meshopt_partitionClusters(
		&cluster_part[0], &cluster_indices[0], c.size_t(len(cluster_indices)),
		&cluster_counts[0], c.size_t(len(cluster_counts)),
		positions, c.size_t(positions_count), c.size_t(mesh.vertex_positions_stride),
		c.size_t(config.partition_size)))

	if config.partition_sort {
		// compute partition points for sorting; any representative point will do, we use last cluster center for simplicity
		partition_point := make([]f32, partition_count * 3)
		defer delete(partition_point)
		for cluster, i in clusters {
			p := int(cluster_part[i]) * 3
			partition_point[p + 0] = cluster.bounds.center[0]
			partition_point[p + 1] = cluster.bounds.center[1]
			partition_point[p + 2] = cluster.bounds.center[2]
		}

		// sort partitions spatially; the output is a remap table from old index (partition id) to new index
		partition_remap := make([]u32, partition_count)
		defer delete(partition_remap)
		meshopt_spatialSortRemap(&partition_remap[0], &partition_point[0], c.size_t(partition_count), c.size_t(size_of(f32) * 3))

		for i in 0 ..< len(clusters) {
			cluster_part[i] = partition_remap[int(cluster_part[i])]
		}
	}

	// count clusters in each partition and compute offsets via prefix sum
	partition_offsets := make([dynamic]int, partition_count + 1)
	for i in 0 ..< len(clusters) {
		partition_offsets[int(cluster_part[i]) + 1] += 1
	}

	for i in 1 ..= partition_count {
		partition_offsets[i] += partition_offsets[i - 1]
	}

	// distribute clusters into contiguous partitions
	write := make([dynamic]int, len(partition_offsets))
	defer delete(write)
	copy(write[:], partition_offsets[:])

	result := make([dynamic]Cluster, len(clusters))
	for i in 0 ..< len(clusters) {
		slot := write[int(cluster_part[i])]
		result[slot] = clusters[i]
		write[int(cluster_part[i])] = slot + 1
	}

	clusters^ = result
	return partition_offsets
}

lock_boundary :: proc(locks: []u8, clusters: []Cluster, group_offsets: []int, cluster_indices: []u32, remap: []u32, vertex_lock: []u8) {
	// for each remapped vertex, use bit 7 as temporary storage to indicate that the vertex has been used by a different group previously
	for i in 0 ..< len(locks) {
		locks[i] &= ~u8(0x81)
	}

	for i in 0 ..< len(group_offsets) - 1 {
		group := clusters[group_offsets[i]:group_offsets[i + 1]]

		// mark all remapped vertices as locked if seen by a prior group
		for &cluster in group {
			for k in 0 ..< cluster.index_count {
				v := int(cluster_indices[cluster.index_offset + k])
				r := int(remap[v])

				locks[r] |= locks[r] >> 7
			}
		}

		// mark all remapped vertices as seen
		for &cluster in group {
			for k in 0 ..< cluster.index_count {
				v := int(cluster_indices[cluster.index_offset + k])
				r := int(remap[v])

				locks[r] |= 1 << 7
			}
		}
	}

	for i in 0 ..< len(locks) {
		r := int(remap[i])

		// consistently lock all vertices with the same position; keep protect bit if set
		locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect)

		if len(vertex_lock) > 0 {
			locks[i] |= vertex_lock[i]
		}
	}
}

SloppyVertex :: struct {
	x, y, z: f32,
	id:      u32,
}

simplify_fallback :: proc(lod: ^[dynamic]u32, mesh: clodMesh, indices: []u32, locks: []u8, target_count: int, error: ^f32) {
	subset := make([]SloppyVertex, len(indices))
	defer delete(subset)
	subset_locks := make([]u8, len(indices))
	defer delete(subset_locks)

	resize(lod, len(indices))

	positions_stride := mesh.vertex_positions_stride / size_of(f32)
	positions := cast([^]f32) mesh.vertex_positions

	// deindex the mesh subset to avoid calling simplifySloppy on the entire vertex buffer (which is prohibitively expensive without sparsity)
	for i in 0 ..< len(indices) {
		v := int(indices[i])
		assert(v < mesh.vertex_count)

		subset[i].x = positions[v * positions_stride + 0]
		subset[i].y = positions[v * positions_stride + 1]
		subset[i].z = positions[v * positions_stride + 2]
		subset[i].id = indices[i]

		subset_locks[i] = locks[v]
		lod[i] = u32(i)
	}

	n := int(meshopt_simplifySloppy(
		&lod[0], &lod[0], c.size_t(len(lod)),
		&subset[0].x, c.size_t(len(subset)), c.size_t(size_of(SloppyVertex)),
		&subset_locks[0], c.size_t(target_count), math.F32_MAX, error))
	resize(lod, n)

	// convert error to absolute
	error^ *= meshopt_simplifyScale(&subset[0].x, c.size_t(len(subset)), c.size_t(size_of(SloppyVertex)))

	// restore original vertex indices
	for i in 0 ..< len(lod) {
		lod[i] = subset[int(lod[i])].id
	}
}

simplify_attributes :: proc(lod: ^[dynamic]u32, mesh: clodMesh, indices: []u32, locks: []u8, target_count: int, options: u32, error: ^f32) -> int {
	return int(meshopt_simplifyWithAttributes(
		&lod[0], &indices[0], c.size_t(len(indices)),
		mesh.vertex_positions, c.size_t(mesh.vertex_count), c.size_t(mesh.vertex_positions_stride),
		mesh.vertex_attributes, c.size_t(mesh.vertex_attributes_stride),
		cast(^f32) raw_data(mesh.attribute_weights), c.size_t(mesh.attribute_count),
		&locks[0], c.size_t(target_count), math.F32_MAX, options, error))
}

simplify :: proc(lod: ^[dynamic]u32, config: clodConfig, mesh: clodMesh, indices: []u32, locks: []u8, target_count: int, error: ^f32) {
	assert(target_count <= len(indices))
	resize(lod, len(indices))

	options: u32 = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute

	if config.simplify_error_clamped {
		options |= meshopt_SimplifyErrorClamped
	}
	if config.simplify_permissive {
		options |= meshopt_SimplifyPermissive
	}
	if config.simplify_regularize {
		options |= meshopt_SimplifyRegularize
	}
	if config.simplify_preserve_folds {
		options |= meshopt_SimplifyPreserveFolds
	}

	resize(lod, simplify_attributes(lod, mesh, indices, locks, target_count, options, error))

	if len(lod) > target_count && config.simplify_fallback_permissive && !config.simplify_permissive {
		resize(lod, simplify_attributes(lod, mesh, indices, locks, target_count, options | meshopt_SimplifyPermissive, error))
	}

	// while it's possible to call simplifySloppy directly, it doesn't support sparsity or absolute error, so we need to do some extra work
	if len(lod) > target_count && config.simplify_fallback_sloppy {
		simplify_fallback(lod, mesh, indices, locks, target_count, error)
		error^ *= config.simplify_error_factor_sloppy // scale error up to account for appearance degradation
	}

	// replace degenerate outputs with the first triangle; this ensures we do not produce empty clusters that break DAG invariants
	if len(lod) == 0 && len(indices) > 0 {
		clear(lod)
		append(lod, indices[0], indices[1], indices[2])
	}

	// optionally limit error by edge length, aiming to remove subpixel triangles even if the attribute error is high
	if config.simplify_error_edge_limit > 0 {
		max_edge_sq: f32 = 0

		for i := 0; i < len(indices); i += 3 {
			va := vertex_pos(mesh, int(indices[i + 0]))
			vb := vertex_pos(mesh, int(indices[i + 1]))
			vc := vertex_pos(mesh, int(indices[i + 2]))

			// compute squared edge lengths
			eab := (va[0] - vb[0]) * (va[0] - vb[0]) + (va[1] - vb[1]) * (va[1] - vb[1]) + (va[2] - vb[2]) * (va[2] - vb[2])
			eac := (va[0] - vc[0]) * (va[0] - vc[0]) + (va[1] - vc[1]) * (va[1] - vc[1]) + (va[2] - vc[2]) * (va[2] - vc[2])
			ebc := (vb[0] - vc[0]) * (vb[0] - vc[0]) + (vb[1] - vc[1]) * (vb[1] - vc[1]) + (vb[2] - vc[2]) * (vb[2] - vc[2])

			emax := max(max(eab, eac), ebc)
			emin := min(min(eab, eac), ebc)

			// we prefer using min edge length to reduce the number of triangles <1px thick, but need some stopgap for thin and long triangles like wires
			max_edge_sq = max(max_edge_sq, max(emin, emax / 4))
		}

		// adjust the error to limit it for dense clusters based on edge lengths
		error^ = min(error^, math.sqrt(max_edge_sq) * config.simplify_error_edge_limit)
	}
}

output_group :: proc(config: clodConfig, mesh: clodMesh, group: []Cluster, cluster_indices: []u32, simplified: clodBounds, depth: int, output_context: rawptr, output_callback: clodOutput) -> int {
	group_clusters := make([]clodCluster, len(group))
	defer delete(group_clusters)

	for cluster, i in group {
		result := &group_clusters[i]

		result.refined = cluster.refined
		if config.optimize_bounds && cluster.refined != -1 {
			result.bounds = bounds_compute(mesh, cluster_indices[cluster.index_offset:cluster.index_offset + cluster.index_count], cluster.bounds.error)
		} else {
			result.bounds = cluster.bounds
		}
		result.indices = cluster_indices[cluster.index_offset:cluster.index_offset + cluster.index_count]
		result.vertex_count = cluster.vertices
	}

	if output_callback != nil {
		return output_callback(output_context, clodGroup{depth = depth, simplified = simplified}, group_clusters)
	}
	return -1
}

edge_lookup :: proc(table: []u64, key: u64) -> ^u64 {
	mask := len(table) - 1
	h := int(((key * 0x9E3779B97F4A7C15) >> 32) & u64(mask))

	for table[h] != ~u64(0) && table[h] != key {
		h = (h + 1) & mask
	}

	return &table[h]
}

boundary_area :: proc(mesh: clodMesh, indices: []u32, locks: []u8, remap: []u32, table: []u64) -> f32 {
	for i in 0 ..< len(table) {
		table[i] = ~u64(0)
	}

	for i := 0; i < len(indices); i += 3 {
		for e := 0; e < 3; e += 1 {
			a := int(remap[int(indices[i + e])])
			b := int(remap[int(indices[i + (e == 2 ? 0 : e + 1)])])
			id := (u64(a) << 32) | u64(b)
			edge_lookup(table, id)^ = id
		}
	}

	area: f32 = 0

	for i := 0; i < len(indices); i += 3 {
		border := false

		for e := 0; e < 3; e += 1 {
			a := int(remap[int(indices[i + e])])
			b := int(remap[int(indices[i + (e == 2 ? 0 : e + 1)])])
			id := (u64(b) << 32) | u64(a)

			// we only track border triangles - at least one edge should be unlocked and open (the reverse edge does not exist)
			border = border || ((locks[a] & locks[b] & 1) == 0 && edge_lookup(table, id)^ == ~u64(0))
		}

		if !border {
			continue
		}

		va := vertex_pos(mesh, int(indices[i + 0]))
		vb := vertex_pos(mesh, int(indices[i + 1]))
		vc := vertex_pos(mesh, int(indices[i + 2]))

		ab := [3]f32{vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]}
		ac := [3]f32{vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]}
		nt := [3]f32{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]}

		area += math.sqrt(nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]) * 0.5
	}

	return area
}

dilate_borders :: proc(mesh: clodMesh, merged: []u32, simplified: []u32, locks: []u8, remap: []u32, bounds: ^clodBounds, dilate_offsets: []f32, edge_table: ^[dynamic]u64) {
	// hash table is sized to be able to track all edges
	table_size := 16
	for table_size < len(merged) + len(merged) / 2 {
		table_size *= 2
	}

	resize(edge_table, table_size)
	edge := edge_table^[:]

	old_area := boundary_area(mesh, merged, locks, remap, edge)
	if old_area == 0 {
		return
	}

	// limit expansion to cases of moderate area shrinkage; it may be unsafe to dilate otherwise
	new_area := boundary_area(mesh, simplified, locks, remap, edge)
	if new_area >= old_area || new_area < old_area / 4 {
		return
	}

	perimeter: f32 = 0
	positions := cast([^]f32) mesh.vertex_positions
	positions_stride := mesh.vertex_positions_stride / size_of(f32)

	for i := 0; i < len(simplified); i += 3 {
		for e := 0; e < 3; e += 1 {
			a := int(remap[int(simplified[i + e])])
			b := int(remap[int(simplified[i + (e == 2 ? 0 : e + 1)])])
			id := (u64(b) << 32) | u64(a)

			// the border check must match the one in boundaryArea (we also rely on table being filled by boundaryArea to avoid extra work)
			if (locks[a] & locks[b] & 1) != 0 || edge_lookup(edge, id)^ != ~u64(0) {
				continue
			}

			va := vertex_pos(mesh, int(simplified[i + e]))
			vb := vertex_pos(mesh, int(simplified[i + (e == 2 ? 0 : e + 1)]))
			vc := vertex_pos(mesh, int(simplified[i + (e + 2) % 3]))

			ev := [3]f32{vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]}
			el := math.sqrt(ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2])

			// compute edge normal by projecting C onto AB
			cv := [3]f32{va[0] - vc[0], va[1] - vc[1], va[2] - vc[2]}
			cp := (cv[0] * ev[0] + cv[1] * ev[1] + cv[2] * ev[2]) / (el > 0 ? el * el : 1)
			nv := [3]f32{cv[0] - ev[0] * cp, cv[1] - ev[1] * cp, cv[2] - ev[2] * cp}
			nl := math.sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2])

			// accumulate edge normals weighted by edge length; we will renormalize by perimeter before applying
			ns := nl > 0 ? el / nl : 0

			if (locks[a] & 1) == 0 {
				dilate_offsets[a * 4 + 0] += nv[0] * ns
				dilate_offsets[a * 4 + 1] += nv[1] * ns
				dilate_offsets[a * 4 + 2] += nv[2] * ns
				dilate_offsets[a * 4 + 3] += el
			}
			if (locks[b] & 1) == 0 {
				dilate_offsets[b * 4 + 0] += nv[0] * ns
				dilate_offsets[b * 4 + 1] += nv[1] * ns
				dilate_offsets[b * 4 + 2] += nv[2] * ns
				dilate_offsets[b * 4 + 3] += el
			}

			perimeter += el
		}
	}

	distance := perimeter > 0 ? (old_area - new_area) / perimeter : 0

	for i := 0; i < len(simplified); i += 1 {
		r := int(remap[int(simplified[i])])
		if locks[r] & 1 != 0 {
			continue
		}

		si := int(simplified[i]) * positions_stride
		ri := r * positions_stride

		n := cast([^]f32) &dilate_offsets[r * 4]

		if n[3] > 0 {
			// we average the normals and then divide by length twice: once to normalize, and once to extend the offset into a miter
			nx := n[0] / n[3]
			ny := n[1] / n[3]
			nz := n[2] / n[3]
			nn := nx * nx + ny * ny + nz * nz
			ns := distance / (nn > 0.15 ? nn : 0.15)

			positions[ri + 0] += nx * ns
			positions[ri + 1] += ny * ns
			positions[ri + 2] += nz * ns
			n[0] = 0
			n[1] = 0
			n[2] = 0
			n[3] = 0

			// expand group bounds to accommodate the offset vertex
			dx := positions[ri + 0] - bounds.center[0]
			dy := positions[ri + 1] - bounds.center[1]
			dz := positions[ri + 2] - bounds.center[2]
			bounds.radius = max(bounds.radius, math.sqrt(dx * dx + dy * dy + dz * dz))
		}

		// copy dilated positions back to all referencing wedges from the canonical copy updated above; no-op for vertices that weren't dilated
		positions[si + 0] = positions[ri + 0]
		positions[si + 1] = positions[ri + 1]
		positions[si + 2] = positions[ri + 2]
	}
}

merge_nodes :: proc(nodes: []clodNode, offset: int, count: int) -> clodNode {
	result := clodNode{}
	result.bounds = bounds_merge(&nodes[offset].bounds, count, size_of(clodNode))
	result.group = -1
	result.child_offset = offset
	result.child_count = count
	return result
}
