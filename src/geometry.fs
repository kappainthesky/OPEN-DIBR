// This geometry shader is necessary to delete elongated triangles, i.e. triangles that connect foreground to background objects

#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;


in vs_out
{
    vec2 TexCoord;
	float angle;
	float inputDepth;
	float outputDepth;
	vec4 worldPosition;
}vertices[];

out fs_in
{
    vec2 TexCoord;
	float angle;
	float outputDepth;
}frag;

uniform vec2 near_far;

uniform sampler2D previousFBOAngleAndDepthTex;

uniform float isFirstInput;
uniform float depth_diff_threshold_fragment;

uniform float triangle_deletion_factor;
uniform float triangle_deletion_margin; 


void main()
{
	// Discard triangles that contain invalid / hole vertices
	if (vertices[0].worldPosition.w < 0.0 || vertices[1].worldPosition.w < 0.0 || vertices[2].worldPosition.w < 0.0) {
		return;
	}

	float d0 = vertices[0].inputDepth;
	float d1 = vertices[1].inputDepth;
	float d2 = vertices[2].inputDepth;

	// Check bounds against near and far clipping planes
	if (d0 < near_far[0] || d1 < near_far[0] || d2 < near_far[0] ||
	    d0 > near_far[1] || d1 > near_far[1] || d2 > near_far[1]) {
		return;
	}

	// Discard triangles that connect vertices with very different depth values (foreground-to-background stretching)
	float min_depth = min(d0, min(d1, d2));
	float max_depth = max(d0, max(d1, d2));
	float largest_depth_diff = max_depth - min_depth;

	float estimated_error = triangle_deletion_factor * (max_depth - near_far[0]) * (max_depth - near_far[0]);

	// Robust depth jump threshold:
	// For 16-bit depth sensors (RealSense), use calibrated depth-dependent discontinuity rejection
	// For 8-bit datasets, preserve legacy error scaling
	float thresh = triangle_deletion_margin * (0.040f + 0.035f * pow(max(0.1f, max_depth), 1.5f));
	if (triangle_deletion_factor > 0.0001f) {
		thresh = triangle_deletion_margin * estimated_error + 0.02f;
	}

	if (largest_depth_diff < thresh) { 
		for (int i = 0; i < 3; i++) {
			frag.TexCoord = vertices[i].TexCoord;
			frag.angle = vertices[i].angle;
			frag.outputDepth = vertices[i].outputDepth;
			gl_Position = gl_in[i].gl_Position;
			EmitVertex();
		}
		EndPrimitive();
	}
}
