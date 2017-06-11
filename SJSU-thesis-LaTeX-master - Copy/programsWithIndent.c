if(threadIdx.x >= 54) {
	GPU_results[0].num_strings = const_num_strings;

	for(int i=0;i<const_num_strings;i++) {
		int patLen = const_indexes[2*i+1] - const_indexes[2*i];
		//This condition checks if the pattern length is < packet length
		if(threadIdx.x<=256-patLen) {
			int hy,j;
			for(hy=j=0;j<patLen;j++) {
				if((j+threadIdx.x) >= 256) goto B;
					hy = (hy * 256 + elements[j+threadIdx.x].packet) % 997;
				}
			if(hy == const_patHash[i] && memCmpDev<T>(elements,const_pattern,const_indexes,i,threadIdx.x,patLen) == 0)  {
				GPU_results[blockIdx.x].maliciousPayload = 1;
				GPU_results[blockIdx.x].signatureNumber = i; 
				d_result[i]=1;
			} 
		}
}		


	if(threadIdx.x>=54)
	{
		int pos=threadIdx.x;
		char ch = elements[pos++].packet;
		int chint = ch & 0x000000FF; 
		int nextState = stateszero[chint];
		if(nextState!=0) {
			if(d_output[nextState] > 0) result[blockIdx.x] = d_output[nextState];
		while(nextState!=0 && pos<256) {
			ch = elements[pos++].packet;
			chint = ch & 0x000000FF;
			nextState = gotofn[nextState*256 + chint];
			if(d_output[nextState] > 0) result[blockIdx.x] = d_output[nextState];
		}
	}
	}

	if (shift == 0) {
		hash2 = elements[threadIdx.x - m + 1].packet & 0x000000FF;
		hash2 <<= 2;
		hash2 += elements[threadIdx.x - m + 2].packet & 0x000000FF;
		//For every pattern with the same suffix as the text
		for (int i = 0; i < d_PREFIX_size[hash1]; i++) {	
		//If the prefix of the pattern matches that of the text
		if (hash2 == d_PREFIX_value[hash1 * prefixPitch + i]) {
			int patIndex = d_PREFIX_index[hash1* prefixPitch + i];
			int starttxt = threadIdx.x - m + 1 + 2;
			int startpat = d_stridx[2*patIndex] + 2;
			int endpat = d_stridx[2*patIndex+1];
			//memcmp implementation
			while(elements[starttxt].packet!='\0' && startpat < endpat) {
				if(elements[starttxt++].packet!=d_pattern[startpat++]) return;
			}
			if(startpat >= endpat) { 
				printf("The pattern exists %d\n", patIndex);
				GPU_results[blockIdx.x].maliciousPayload = 1;
				result[blockIdx.x] = patIndex;
			}
		}