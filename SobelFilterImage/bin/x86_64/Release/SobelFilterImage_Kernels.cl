/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void erode(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float prog_binaryzacji = 145.0;
	bool wynik_erozji = 0;	
	int size_mask=3;
	float4 pixel_context = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	float3 RGB=(float3)(pixel_context.x,pixel_context.y,pixel_context.z);
	float3 conv_vector=(float3)(0.2989,0.5870,0.1140);
	float value = dot(RGB,conv_vector);
	bool pixel_context_bool = step(prog_binaryzacji,value);
	for (int i=-size_mask/2;i<=size_mask/2;i++){
		for (int j=-size_mask/2;j<=size_mask/2;j++){
			pixel_context = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x+i, coord.y+j)));
			float3 RGB=(float3)(pixel_context.x,pixel_context.y,pixel_context.z);
			float3 conv_vector=(float3)(0.2989,0.5870,0.1140);
			float value = dot(RGB,conv_vector);
			pixel_context_bool=pixel_context_bool && (bool)(step(prog_binaryzacji,value));
			}
		}
	write_imageui(outputImage, coord,255*(uint)pixel_context_bool);			
}

__kernel void erode_diff(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float prog_binaryzacji = 145.0;
	bool wynik_erozji = 0;	
	int size_mask=3;
	float4 pixel_context = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	float og_pixel = (convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)))).y;
	bool pixel_context_bool = (bool) pixel_context.y;
	for (int i=-size_mask/2;i<=size_mask/2;i++){
		for (int j=-size_mask/2;j<=size_mask/2;j++){
			pixel_context = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x+i, coord.y+j)));
			pixel_context_bool=pixel_context_bool && (bool)pixel_context.y;
			}
		}
	uint wynik = (uint) og_pixel - 255*(uint) pixel_context_bool ;
	write_imageui(outputImage, coord, wynik);			
}
