// blending of current output image with the previous (if !isFirstInput)

#version 330 core
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec2 FragAngleAndDepth;

in fs_in
{
    vec2 TexCoord;
	float angle;
	float outputDepth;
}frag;

uniform float width;
uniform float height;
uniform float out_width;
uniform float out_height;
uniform float chroma_offset;

uniform float blendingThreshold;
uniform float image_border_threshold_fragment;
uniform float depth_diff_threshold_fragment;

uniform sampler2D colorTex;
uniform sampler2D previousFBOColorTex;
uniform sampler2D previousFBOAngleAndDepthTex;

uniform float isFirstInput;
uniform float isYCbCr;
uniform float convertYCbCrToRGB;
uniform float debugMode; // 0=normal, 3=depth heatmap, 4=invalid mask, 5=stereo difference
uniform vec2 near_far;

vec3 jet_colormap(float val) {
    val = clamp(val, 0.0, 1.0);
    float r = clamp(1.5 - abs(4.0 * val - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * val - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * val - 1.0), 0.0, 1.0);
    return vec3(r, g, b);
}

void main()
{
	// SAMPLE colorTex
	// -------------------------------------
	if(isYCbCr > 0.5f){
		// color tex is YUV NV12
		vec2 clamped_coord = clamp(frag.TexCoord, vec2(0.0f), vec2(1.0f));
		vec2 texcoord_Y = vec2(clamped_coord.x, clamped_coord.y * height / (height * 1.5f + chroma_offset));
		float Cb_x = (floor(floor(clamped_coord.x * width) / 2.0f) * 2.0f + 0.5f) / width;
		float Cr_x = (floor(floor(clamped_coord.x * width) / 2.0f) * 2.0f + 1.5f) / width;
		
		float min_chroma_y = (height + chroma_offset + 0.5f) / (height * 1.5f + chroma_offset);
		float max_chroma_y = (height * 1.5f + chroma_offset - 0.5f) / (height * 1.5f + chroma_offset);
		float Cb_Cr_y = (floor(floor(clamped_coord.y * height) / 2.0f) + 0.5f + height + chroma_offset) / (height * 1.5f + chroma_offset);
		Cb_Cr_y = clamp(Cb_Cr_y, min_chroma_y, max_chroma_y);
	
		float Y = texture(colorTex, texcoord_Y).r;
		float Cb = texture(colorTex, vec2(Cb_x, Cb_Cr_y)).r;
		float Cr = texture(colorTex, vec2(Cr_x, Cb_Cr_y)).r;

		// Fallback to neutral chroma (128/255 = 0.50196) if sampling out-of-bounds or border padding
		if (frag.TexCoord.x < 0.0f || frag.TexCoord.x > 1.0f || frag.TexCoord.y < 0.0f || frag.TexCoord.y > 1.0f) {
			Cb = 128.0f / 255.0f;
			Cr = 128.0f / 255.0f;
		}

		FragColor = vec4(Y, Cb, Cr, 1);

		if(convertYCbCrToRGB > 0.5f){
			// CONVERT TO RGB
			// --------------
			float r = Y + 1.370705*(Cr - 128.0f / 255.0f);
			float g = Y - 0.698001 *(Cr - 128.0f / 255.0f) - 0.337633*(Cb - 128.0f / 255.0f);
			float b = Y + 1.732446*(Cb - 128.0f / 255.0f);
			FragColor = vec4(clamp(r, 0.0f, 1.0f), clamp(g, 0.0f, 1.0f), clamp(b, 0.0f, 1.0f), 1);
		}
	}
	else {
		// color tex is RGB
		FragColor = texture(colorTex, frag.TexCoord);
	}
	
	FragAngleAndDepth = vec2(frag.angle, frag.outputDepth);

	// LET THE FragAngle INCREASE WHEN CLOSE TO THE BORDER OF THE INPUT IMAGE
	float col = frag.TexCoord.x * width;
	float row = frag.TexCoord.y * height;

	float distance_to_image_border = min(min(min(col, width-col), row), height-row);
	if(distance_to_image_border < image_border_threshold_fragment){
		FragAngleAndDepth.x = FragAngleAndDepth.x + 4.0f * blendingThreshold * (1.0f-distance_to_image_border / image_border_threshold_fragment);
	}

	if(isFirstInput < 0.5f){
		// MULTI-VIEW VISIBILITY & FUSION PASS
		vec2 TexCoordPreviousTex = vec2(gl_FragCoord.x / out_width, gl_FragCoord.y / out_height);
		vec4 previous_color = texture(previousFBOColorTex, TexCoordPreviousTex);
		vec2 previous_angle_and_depth = texture(previousFBOAngleAndDepthTex, TexCoordPreviousTex).xy;
		
		float current_depth = frag.outputDepth;
		float previous_depth = previous_angle_and_depth.y;
		float depth_epsilon = max(0.025f, depth_diff_threshold_fragment); // geometric surface tolerance

		float blendfactor = 0.0f; // weight of current input image (Camera 1 / Right)

		if (previous_depth > 9000.0f) {
			// Case 1: Previous camera had hole/disocclusion -> Current camera fills 100%
			blendfactor = 1.0f;
		}
		else if (current_depth > previous_depth + depth_epsilon) {
			// Case 2: Current camera is behind previous foreground surface -> Discard occluded geometry
			discard;
		}
		else if (current_depth < previous_depth - depth_epsilon) {
			// Case 3: Current camera is strictly in front -> Overwrite background with 100% weight (no ghosting)
			blendfactor = 1.0f;
		}
		else {
			// Case 4: Both cameras observe the SAME physical surface (depth consistent)
			// Select dominant camera based on viewing angle, with smooth Hermite transition near the seam
			float current_angle = frag.angle;
			float previous_angle = previous_angle_and_depth.x;
			float angle_diff = previous_angle - current_angle; // positive when current camera has smaller angle (better)

			float seam_width = max(0.002f, blendingThreshold);
			float t = clamp((angle_diff / (2.0f * seam_width)) + 0.5f, 0.0f, 1.0f);
			blendfactor = smoothstep(0.0f, 1.0f, t);

			// Debug Mode 4: Depth Consistency Error Heatmap (|Z_left - Z_right| in mm)
			if (debugMode > 4.5f && debugMode < 5.5f) {
				float z_diff_mm = abs(current_depth - previous_depth) * 1000.0f;
				float norm_err = clamp(z_diff_mm / 100.0f, 0.0f, 1.0f); // 0mm = blue, 50mm = green, 100mm = red
				FragColor = vec4(jet_colormap(norm_err), 1.0f);
			}
			// Debug Mode 3: Camera Ownership / Z-Buffer Winner Map (Left = Red tint, Right = Cyan tint)
			else if (debugMode > 2.5f && debugMode < 3.5f) {
				if (blendfactor > 0.5f) {
					FragColor = vec4(0.2f * FragColor.r, FragColor.g, FragColor.b, 1.0f); // Cyan tint for Right camera
				} else {
					FragColor = vec4(previous_color.r, 0.2f * previous_color.g, 0.2f * previous_color.b, 1.0f); // Red tint for Left camera
				}
			}
		}

		if (debugMode < 2.5f || debugMode > 5.5f) {
			FragColor = blendfactor * FragColor + (1.0f - blendfactor) * previous_color;
			FragAngleAndDepth = blendfactor * FragAngleAndDepth + (1.0f - blendfactor) * previous_angle_and_depth;
		}
	}

	// DEBUG OVERLAYS
	if (debugMode > 5.5f && debugMode < 6.5f) {
		// Mode 6: Depth Colormap Heatmap
		float norm_d = (frag.outputDepth - near_far.x) / max(0.001f, (near_far.y - near_far.x));
		FragColor = vec4(jet_colormap(norm_d), 1.0f);
	}
	else if (debugMode > 6.5f && debugMode < 7.5f) {
		// Mode 7: Invalid Depth Mask
		if (frag.outputDepth > near_far.y - 0.1f || frag.outputDepth < near_far.x + 0.01f) {
			FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f); // Red for holes/invalid
		} else {
			FragColor = vec4(0.2f, 0.8f, 0.2f, 1.0f); // Green for valid
		}
	}
	else if (debugMode > 7.5f) {
		// Mode 8: Stereo Anaglyph Overlap (Left=Red, Right=Cyan in target camera space)
		if (isFirstInput > 0.5f) {
			FragColor = vec4(FragColor.r, 0.0f, 0.0f, 1.0f);
		} else {
			FragColor = vec4(0.0f, FragColor.g, FragColor.b, 1.0f);
		}
	}
}