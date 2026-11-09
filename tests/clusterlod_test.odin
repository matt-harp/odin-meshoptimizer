package main

import "core:fmt"
import "core:math"
import "core:os"
import meshopt "../"

/*
 * Unit tests for the native Odin port of clusterlod (clusterlod.odin).
 *
 * These exercise:
 *   - clodDefaultConfig / clodDefaultConfigRT defaults
 *   - clodLocalIndices round-trip
 *   - clodBuild end-to-end on a regular grid (cluster LOD invariants)
 *   - clodBuildHierarchy / clodBuildHierarchyBound structural invariants
 */

failures := 0

check :: proc(cond: bool, name: string) {
	if !cond {
		fmt.eprintln("FAILED:", name)
		failures += 1
	} else {
		fmt.println("ok:", name)
	}
}

// A captured snapshot of a cluster as seen by the clodBuild output callback.
CapturedCluster :: struct {
	refined:     int,
	bounds:      meshopt.clodBounds,
	indices:     [dynamic]u32,
	vertex_count: int,
}

CapturedGroup :: struct {
	depth:      int,
	simplified: meshopt.clodBounds,
	clusters:   [dynamic]CapturedCluster,
}

GroupState :: struct {
	groups: [dynamic]CapturedGroup,
}

// Callback used by the clodBuild test; records every emitted group and returns
// the group id, which clodBuild stores as clodCluster.refined on the next level.
group_callback :: proc(ctx: rawptr, group: meshopt.clodGroup, clusters: []meshopt.clodCluster) -> int {
	state := cast(^GroupState) ctx

	cg := CapturedGroup{depth = group.depth, simplified = group.simplified}
	for c in clusters {
		cc := CapturedCluster{
			refined = c.refined,
			bounds = c.bounds,
			vertex_count = c.vertex_count,
		}
		cc.indices = make([dynamic]u32, len(c.indices))
		copy(cc.indices[:], c.indices)
		append(&cg.clusters, cc)
	}
	append(&state.groups, cg)

	return len(state.groups) - 1
}

// Builds an n x n regular grid mesh (two triangles per cell, CCW winding).
build_grid_mesh :: proc(n: int) -> (indices: []u32, positions: []f32, vertex_count: int) {
	vertex_count = n * n

	positions = make([]f32, vertex_count * 3)
	for y in 0 ..< n {
		for x in 0 ..< n {
			i := y * n + x
			positions[i * 3 + 0] = f32(x)
			positions[i * 3 + 1] = f32(y)
			positions[i * 3 + 2] = 0
		}
	}

	indices = make([]u32, (n - 1) * (n - 1) * 6)
	k := 0
	for y in 0 ..< n - 1 {
		for x in 0 ..< n - 1 {
			a := u32(y * n + x)
			b := u32(y * n + x + 1)
			c := u32((y + 1) * n + x)
			d := u32((y + 1) * n + x + 1)

			indices[k + 0] = a
			indices[k + 1] = c
			indices[k + 2] = b
			indices[k + 3] = b
			indices[k + 4] = c
			indices[k + 5] = d
			k += 6
		}
	}

	return
}

test_default_config :: proc() {
	c := meshopt.clodDefaultConfig(64)
	check(c.max_vertices == 64, "default max_vertices")
	check(c.min_triangles == 21, "default min_triangles")
	check(c.max_triangles == 64, "default max_triangles")
	check(c.partition_spatial, "default partition_spatial")
	check(c.partition_size == 16, "default partition_size")
	check(!c.cluster_spatial, "default cluster_spatial")
	check(c.cluster_split_factor == 2.0, "default cluster_split_factor")
	check(c.simplify_ratio == 0.5, "default simplify_ratio")
	check(c.simplify_permissive, "default simplify_permissive")
	check(c.simplify_fallback_sloppy, "default simplify_fallback_sloppy")
	check(c.optimize_clusters, "default optimize_clusters")

	rt := meshopt.clodDefaultConfigRT(64)
	check(rt.min_triangles == 16, "RT min_triangles")
	check(rt.max_vertices == 128, "RT max_vertices")
	check(rt.cluster_spatial, "RT cluster_spatial")
	check(rt.cluster_fill_weight == 0.5, "RT cluster_fill_weight")
	check(rt.simplify_permissive, "RT inherits simplify_permissive")
}

test_local_indices :: proc() {
	indices := [?]u32{5, 7, 2}
	vertices := make([]u32, 3)
	triangles := make([]u8, 3)

	unique := meshopt.clodLocalIndices(vertices[:], triangles[:], indices[:])
	check(unique == 3, "clodLocalIndices unique count")

	// Round-trip: vertices[triangles[i]] must reproduce the original indices.
	ok := true
	for i in 0 ..< 3 {
		if vertices[int(triangles[i])] != indices[i] {
			ok = false
		}
	}
	check(ok, "clodLocalIndices round-trip")
}

test_build_grid :: proc() {
	indices, positions, vertex_count := build_grid_mesh(12)
	defer delete(indices)
	defer delete(positions)

	mesh := meshopt.clodMesh{
		indices = indices,
		vertex_count = vertex_count,
		vertex_positions = &positions[0],
		vertex_positions_stride = 3 * size_of(f32),
	}

	config := meshopt.clodDefaultConfig(64)
	state := GroupState{}
	defer {
		for g in state.groups {
			for c in g.clusters {
				delete(c.indices)
			}
			delete(g.clusters)
		}
		delete(state.groups)
	}

	meshopt.clodBuild(config, mesh, &state, group_callback)

	check(len(state.groups) >= 1, "clodBuild produces groups")

	// Depths must be emitted in non-decreasing order.
	depth_ok := true
	for i in 1 ..< len(state.groups) {
		if state.groups[i].depth < state.groups[i - 1].depth {
			depth_ok = false
		}
	}
	check(depth_ok, "group depths are non-decreasing")

	check(len(state.groups) > 0 && state.groups[0].depth == 0, "first group is depth 0")

	// The last emitted group is terminal (error == F32_MAX).
	last := state.groups[len(state.groups) - 1]
	check(last.simplified.error == math.F32_MAX, "terminal group error is F32_MAX")

	// Per-cluster invariants.
	cluster_ok := true
	for g, gi in state.groups {
		for c in g.clusters {
			if len(c.indices) == 0 || len(c.indices) % 3 != 0 {
				cluster_ok = false
			}
			if c.vertex_count <= 0 || c.vertex_count > len(c.indices) {
				cluster_ok = false
			}
			for idx in c.indices {
				if int(idx) >= vertex_count {
					cluster_ok = false
				}
			}
			// refined must either be -1 (original geometry) or a previously emitted group id.
			if c.refined != -1 && (c.refined < 0 || c.refined >= gi) {
				cluster_ok = false
			}
		}
	}
	check(cluster_ok, "cluster invariants hold")

	// Depth 0 clusters must reference original geometry.
	root_refined_ok := true
	for c in state.groups[0].clusters {
		if c.refined != -1 {
			root_refined_ok = false
		}
	}
	check(root_refined_ok, "depth 0 clusters have refined == -1")
}

test_build_tiny :: proc() {
	indices := [?]u32{0, 1, 2}
	positions := [?]f32{0, 0, 0, 1, 0, 0, 0, 1, 0}

	mesh := meshopt.clodMesh{
		indices = indices[:],
		vertex_count = 3,
		vertex_positions = &positions[0],
		vertex_positions_stride = 3 * size_of(f32),
	}

	config := meshopt.clodDefaultConfig(64)
	state := GroupState{}
	defer {
		for g in state.groups {
			for c in g.clusters {
				delete(c.indices)
			}
			delete(g.clusters)
		}
		delete(state.groups)
	}

	meshopt.clodBuild(config, mesh, &state, group_callback)

	check(len(state.groups) == 1, "tiny mesh produces a single group")
	check(len(state.groups) == 1 && state.groups[0].depth == 0, "tiny group is depth 0")
	check(len(state.groups) == 1 && state.groups[0].simplified.error == math.F32_MAX, "tiny group is terminal")
}

test_hierarchy :: proc() {
	groups := []meshopt.clodGroup{
		{depth = 0, simplified = {center = {0, 0, 0}, radius = 1, error = 0}},
		{depth = 0, simplified = {center = {4, 0, 0}, radius = 1, error = 0}},
		{depth = 0, simplified = {center = {8, 0, 0}, radius = 1, error = 0}},
		{depth = 1, simplified = {center = {2, 0, 0}, radius = 3, error = 1}},
		{depth = 1, simplified = {center = {6, 0, 0}, radius = 3, error = 1}},
	}
	level_count := 2
	node_width := 2

	bound := meshopt.clodBuildHierarchyBound(len(groups), node_width, level_count)
	nodes := make([]meshopt.clodNode, bound)
	defer delete(nodes)

	count := meshopt.clodBuildHierarchy(nodes[:], groups[:], node_width, level_count)
	check(count > level_count, "hierarchy builds internal nodes")
	check(count <= bound, "hierarchy node count respects bound")

	// Count leaves; they should exactly correspond to the input groups.
	leaf_count := 0
	struct_ok := true
	for i in 0 ..< count {
		node := nodes[i]
		if node.child_count == 0 {
			if node.group < 0 || node.group >= len(groups) {
				struct_ok = false
			}
			leaf_count += 1
		} else {
			if node.group != -1 {
				struct_ok = false
			}
			if node.child_offset + node.child_count > count {
				struct_ok = false
			}
		}
	}
	check(struct_ok, "hierarchy node structure is valid")
	check(leaf_count == len(groups), "hierarchy leaf count equals group count")

	// Roots are stored at nodes[0..level_count-1]; verify each is a valid node.
	roots_ok := true
	for level in 0 ..< level_count {
		node := nodes[level]
		if node.child_count == 0 {
			if node.group < 0 || node.group >= len(groups) || groups[node.group].depth != level {
				roots_ok = false
			}
		} else if node.group != -1 {
			roots_ok = false
		}
	}
	check(roots_ok, "per-level hierarchy roots are valid")
}

main :: proc() {
	test_default_config()
	test_local_indices()
	test_build_tiny()
	test_build_grid()
	test_hierarchy()

	if failures != 0 {
		fmt.eprintln("clusterlod tests FAILED:", failures)
		os.exit(1)
	}
	fmt.println("clusterlod tests OK")
}
