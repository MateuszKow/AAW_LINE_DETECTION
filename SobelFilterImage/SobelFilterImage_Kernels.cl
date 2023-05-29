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


__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR; 

__kernel void erode(__read_only image2d_t inputImage, __write_only image2d_t outputImage, uint prog_binaryzacji)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
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
	// uint wynik = 0 ;
	// if (coord.x == 156 &&	coord.y == 22){
	// 	printf("coord.x: %d, coord.y: %d \n",coord.x,coord.y);
		
	// 	printf("coord.x: %d, coord.y: %d \n",coord.x,coord.y);
	// 	write_imageui(outputImage, coord, convert_uint4((float4)(255,0,0,0)));
	// }
	// else{
	// 	write_imageui(outputImage, coord, wynik);
	// }
	
			
}

__kernel void hough_transform(__read_only image2d_t inputImage, __write_only image2d_t outputImage, uint theta_resolution){
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float PI = 3.14159265358979323846;
	float4 pixel_value = convert_float(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)).x);
	// if (coord.x >= 180 && coord.x <= 210 && coord.y >= 180 && coord.y <= 210){
		if (pixel_value.x == 255){
			for (int theta=-90;theta<90;theta=theta+theta_resolution){
				float rho = coord.x*cos((float)theta*PI/180) + coord.y*sin((float)theta*PI/180);
				float4 neg_pixel = (float4)(255,0,0,0);
				// printf("theta: %d, rho: %f\n",theta,rho);
				int rho_display = (int)rho + 200;
				int theta_display = (int)theta + 90;
				// printf("theta: %d, rho: %d \n",theta_display,rho_display);
				write_imageui(outputImage, (int2)(theta_display,rho_display), convert_uint4(neg_pixel));
			}
		}
		else{
			;
		}
	// }
}

__kernel void hough_transform_buffer(__read_only image2d_t inputImage,  __global int* out, uint theta_resolution, uint width, uint height){
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float PI = 3.14159265358979323846;
	float4 pixel_value = convert_float(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)).x);
	// printf("pixel_value: %f\n",pixel_value.x);
	// if (coord.x >= 180 && coord.x <= 210 && coord.y >= 180 && coord.y <= 210){
	// if ((coord.x == 200 && coord.y == 200)||(coord.x == 201 && coord.y == 200)){
	coord.y= abs(height/2-coord.y)+height/2-1;	
		if (pixel_value.x == 255.0f){
			// printf("coord.x: %d, coord.y: %d \n",coord.x,coord.y);
			// if (coord.x == 155 && coord.y == 123){
			// 	printf("coord.x: %d, coord.y: %d \n",coord.x,coord.y);
			// }
			for (int theta=-90;theta<90;theta=theta+theta_resolution){
				int rho = (int)round(coord.x*cos((float)theta *PI/180) + coord.y*sin((float)theta *PI/180)) + (height+width);
				// printf("theta: %d, rho: %d \n",theta,rho);
				// printf("%d\n",rho);
				// printf("theta: %d, rho: %d, index: %d \n",theta,rho ,(theta + 90) + coord.x*180 + coord.y * 180 * width);
				out[(theta + 90) + coord.x*180 + coord.y * 180 * width ] = rho;
				// if (coord.x == 155 && coord.y == 123)	{
				// printf("theta: %d, rho: %d",theta,rho);
				// }
				
			}
		}

		else{
			for (int theta=-90;theta<90;theta=theta+theta_resolution){
				// printf("theta: %d, rho: %d, index: %d \n",theta,5000 ,(theta + 90) + coord.x*180 + coord.y * 180 * width);
				out[(theta + 90) + coord.x*180 + coord.y * 180 * width]  = 5000;
				}
		}
	// }
}

// __kernel void hough_transform_image3D(__read_only image2d_t inputImage,  __write_only image3d_t outputImage, uint theta_resolution){
// 	int3 coord = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
// 	float PI = 3.14159265358979323846;
// 	float4 pixel_value = convert_float(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)).x);
// 	// // if (coord.x >= 180 && coord.x <= 210 && coord.y >= 180 && coord.y <= 210){
// 	// // if ((coord.x == 200 && coord.y == 200)||(coord.x == 201 && coord.y == 200)){
// 	// 	if (pixel_value.x == 255.0f){
// 	// 		// printf("coord.x: %d, coord.y: %d \n",coord.x,coord.y);
// 	// 		int rho = (int)round(coord.x*cos((float)1 *PI/180) + coord.y*sin((float)1 *PI/180));
// 	// 		write_imagef(outputImage, (int4)(coord.x, coord.y, 1,0), convert_float(rho));
// 	// 		// for (int theta=-90;theta<=90;theta=theta+theta_resolution){
// 	// 		// 	int rho = (int)round(coord.x*cos((float)theta *PI/180) + coord.y*sin((float)theta *PI/180));
// 	// 			// printf("theta: %d, rho: %d \n",theta,rho);
// 	// 			// printf("%d\n",rho);
// 	// 			// printf("theta: %d, rho: %d, index: %d \n",theta,rho ,(theta + 90) + coord.x*180 + coord.y * 180 * width);
// 	// 			// out[(theta + 90) + coord.x*180 + coord.y * 180 * width ] = rho;
// 	// 			// write_imagef(outputImage, (int4)(coord.x, coord.y, theta + 90,0), convert_float(rho));
// 	// 		// }
// 	// 	}
// 	// 	else{
// 	// 		write_imagef(outputImage, (int4)(coord.x, coord.y, 1,0), convert_float(5000));
// 	// 		// for (int theta=-90;theta<=90;theta=theta+theta_resolution){
// 	// 			// printf("theta: %d, rho: %d, index: %d \n",theta,5000 ,(theta + 90) + coord.x*180 + coord.y * 180 * width);
// 	// 			// out[(theta + 90) + coord.x*180 + coord.y * 180 * width]  = 5000;
// 	// 			// write_imagef(outputImage, (int4)(coord.x, coord.y, theta + 90,0), convert_float(5000));
// 	// 			// }
// 	// 	}
// 	// }
// }


// __kernel void accumulator(__read_only image3d_t inputImage, __global int* out){
// 	int3 coord = (int3)(get_global_id(0), get_global_id(1),get_global_id(2));
// 	int rho = (int)read_imagef(inputImage, imageSampler, (int4)(coord.x, coord.y, coord.z,0)).x;
// 	int2 cord_out = (int2)(coord.z, rho);
// 	atomic_inc(&out[coord.z + rho*180]);
// 	// write_imagef(outputImage, cord_out, convert_float4((float4)(255,0,0,0)));
// }

__kernel void accumulator(__global int* in, __global int* out){
	// (theta + 90) + coord.x*180 + coord.y * 180 * width 
	int pos = get_global_id(0);
	int theta = pos % 180;
	int rho = in[pos];
		// printf("theta: %d, rho: %d, index: %d, wartosc przed inkremenetacja %d \n",theta,rho ,pos, out[theta + rho*180]);
	if (rho != 5000){
		// printf("theta: %d, rho: %d, index: %d, wartosc przed inkremenetacja %d \n",theta,rho ,pos, out[theta + rho*180]);
		atomic_add(&out[theta + rho*180],1);
		// printf("theta: %d, rho: %d, index: %d, wartosc po inkremenetacji %d \n",theta,rho ,pos, out[theta + rho*180]);
	}
		// printf("theta: %d, rho: %d, index: %d, wartosc po inkremenetacji %d \n",theta,rho ,pos, out[theta + rho*180]);

}



__kernel void to_display(__global int * in,  __global int* out, uint threshhold, uint width){

	int pos = get_global_id(0);
	// pos = rho*180+theta
	// printf("pos %d\n",pos);
	// if (pos==42897){
	// 	printf("pos %d\n",pos);
	// if(pos == 60840 || pos == 71190 || pos == 92970 || pos == 42660){
	// 	printf("Jestem w pos %d\n",pos);
	// 	printf("wartosc %d\n",in[pos]);
	// }

	if (in[pos]>180){
		// printf("maximum %d\n",in[pos]);
		int rho= pos / 180;
		// printf("pos: %d, width: %d \n",pos,width);
		int theta= pos % 180;
		// printf("theta: %d, rho: %d, pos: %d \n",theta,rho,pos);
		// printf("theta: %d, rho: %d \n",theta,rho);
		out[2*pos] = theta;
		out[2*pos+1] = rho;
	}
	else{
		out[2*pos]=5000;
		out[2*pos+1]=5000;
	}
	// }
}