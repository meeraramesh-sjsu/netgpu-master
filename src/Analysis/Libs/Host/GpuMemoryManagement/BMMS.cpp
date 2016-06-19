#include "BMMS.h"

//Static vars initialization
bool BMMS::work = false;
bitmap_t* BMMS::bitmap = NULL;
uint8_t* BMMS::allocatedMemory = NULL;
unsigned int BMMS::blockSize = 0;
unsigned int BMMS::numOfBlocks = 0;

unsigned int inline divCeil(unsigned int val,unsigned int div)
{
	if(val%div == 0)
		return val/div;
	else
		return val/div+1;
		
}


		
void BMMS::init(unsigned int totalSize,unsigned int minimumBlockSize){

	if(totalSize%minimumBlockSize!=0){
		cerr<<"BMMS: Totalsize must be multiple of minimumBlockSize"<<endl;
		exit(-1);
	}
	blockSize = minimumBlockSize;	
	numOfBlocks = totalSize/blockSize;

	if((bitmap = (bitmap_t*)malloc(sizeof(bitmap_t)*numOfBlocks))==NULL){ //cudaMalloc
		cerr<<"Error malloc bitmap"<<endl; 
		exit(-1);
	}
	memset(bitmap,0,sizeof(bitmap_t)*numOfBlocks);	

	cudaAssert(cudaMalloc((void**)&allocatedMemory,totalSize));//cudaMalloc

	if(allocatedMemory == NULL){
		ABORT("BMMS: Error cudaMalloc Buffer");
	}
	#ifdef _DEBUG
		fprintf(stderr,"BMMS GPU allocation at: %p\n",allocatedMemory);
	#endif	

	//Set work flag
	work = true;	
	cerr<<"[BMMS enabled (total memory size:"<<totalSize<<",block size:"<<minimumBlockSize<<")]"<<endl;

}

int BMMS::findAndAssignPortion(unsigned int blocksRequired){
	unsigned int i,j=0;
	int index=-1;

	for(i=0;i<numOfBlocks;i++){
	
		if(bitmap[i].isUsed==false){
			j++;
		}else{
			j=0;	
		}	

		//Enough blocks
		if(j == blocksRequired){
			index = i-j+1;
			break;
		}
	}

	if(index != -1){

		//MARK
		for(i=0;i<blocksRequired;i++){
			bitmap[index+i].isUsed = true;
			bitmap[index+i].blockStart = index;
			bitmap[index+i].numOfPartitionBlocks = blocksRequired;
		}
			
	}	

	return index;

}

void BMMS::mallocBMMS(void** pointer, unsigned int reqSize){


	if(work){
		*pointer = _mallocBMMS(reqSize);

		#ifdef _DEBUG
			printBitmap();
		#endif
	}else
		cudaAssert(cudaMalloc((void**)pointer,reqSize));//cudaMalloc


}

void* BMMS::_mallocBMMS(unsigned int reqSize)
{
	int pos,blocksRequired; 

	blocksRequired = divCeil(reqSize,blockSize);	
	#ifdef _DEBUG
		cerr<<"BMMS::malloc["<<reqSize<<"]"<<endl<<"num of blocks:"<<numOfBlocks<<", Block size:"<<blockSize<<"-> Required("<<reqSize<<"): "<<blocksRequired<<endl;
	#endif	
	
	pos = findAndAssignPortion(blocksRequired);		

	#ifdef _DEBUG
		cerr<<"num of blocks:"<<numOfBlocks<<", Block size:"<<blockSize<<"-> Required("<<reqSize<<"): "<<blocksRequired<<endl;
		cerr<<"Allocato en ["<<pos<<"]"<<endl;
		fprintf(stderr,"[%d] -> %p\n",pos,allocatedMemory+(pos*blockSize));
	#endif

	if(pos==-1){
		cerr<<"BMMS->WARN:Not enough memory or could not allocate(frag.):"<<reqSize<<"bytes"<<endl;
		return NULL;
	}else{
		return (void*)(allocatedMemory+(pos*blockSize));
	}
}

void BMMS::freeBMMS(void* pointer)
{

	unsigned int i;
	int pos;

	if(pointer == NULL)	
		return;
	
	if(work){
	
		//TODO: check pointer is the start position of the block
		pos = ((uint8_t*)pointer-(uint8_t*)allocatedMemory)/blockSize;
		if(pos >=(int)numOfBlocks || pos<0)
			return;

		#ifdef _DEBUG
			cerr<<"BMMS::Free ["<<pos<<"]"<<endl;
		#endif

		if(bitmap[pos].isUsed){
			unsigned int numOfPartitions = bitmap[pos].numOfPartitionBlocks;	
			for(i=0;i<numOfPartitions;i++){
				bitmap[pos+i].isUsed = false;
				bitmap[pos+i].blockStart = 0;
				bitmap[pos+i].numOfPartitionBlocks = 0;
			
			}
		}
	}else
{		
cudaAssert(cudaFree(pointer));//cudaFree
pointer=NULL;
}
}

void BMMS::printBitmap(){
	cerr<<endl<<"Bitmap:"<<endl;
	cerr<<"---------------------"<<endl;
		
	for(unsigned int i=0;i<numOfBlocks;i++){
		if(bitmap[i].isUsed)
			cerr<<"("<<i<<")[X,"<<bitmap[i].blockStart<<","<<bitmap[i].numOfPartitionBlocks<<"]";
		//else
		//	cerr<<"("<<i<<")[,,]";
		//if(i%10 == 0 && i != 0)
		//	cerr<<endl;
	}
	cerr<<endl<<endl;
}
