package main

import "core:c"
import "core:fmt"
import "core:os"
import meshopt "../"

/*
 * Canary: link against the freshly built meshoptimizer library and make sure a
 * real call runs and produces the expected result. If the foreign import can't
 * find/load the library, or the symbol is wrong, this fails to build; if the
 * result is wrong, it exits non-zero.
 */
main :: proc() {
	// A single triangle with three unique vertices.
	indices := [?]u32{0, 1, 2}

	stats := meshopt.meshopt_analyzeVertexCache(
		&indices[0],
		c.size_t(len(indices)),
		c.size_t(3),
		16, // cache_size
		32, // warp_size
		3,  // primgroup_size
	)

	fmt.println("vertices_transformed:", stats.vertices_transformed)
	fmt.println("acmr:", stats.acmr)
	fmt.println("atvr:", stats.atvr)

	if stats.vertices_transformed != 3 {
		fmt.eprintln("canary FAILED: expected 3 vertices transformed")
		os.exit(1)
	}

	fmt.println("canary OK: meshoptimizer linked and ran successfully")
}
