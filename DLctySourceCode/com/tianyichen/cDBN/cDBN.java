package com.tianyichen.cDBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.cRBM.cRBM;
import com.tianyichen.layer.OutputLayer;
import com.tianyichen.layer.SigmoidLayer;
import com.tianyichen.utils.otherTools;

public class cDBN {
	
	private Map<Integer, SigmoidLayer> sigmoidlayers;
	private Map<Integer, cRBM> crbmlayers;
	private List<Integer> layerSizes;
	private Map<Integer, String> layerTypes;
	private OutputLayer outputlayer;
	private Map<Integer, RealMatrix> weights;
	private Map<Integer, RealMatrix> bias;
	private int pretraining_epoch;
	
	
	public cDBN(){}
	
	public cDBN(int inputSize,int[] hiddenSizes,int outputSize,int K, int pretraining_epoch,double alpha){
		
		// set layer sizes of DBN
		this.generateLayerSizes(inputSize, hiddenSizes, outputSize);
		this.sigmoidlayers=new HashMap<Integer, SigmoidLayer>();
		this.crbmlayers=new HashMap<Integer, cRBM>();
		this.pretraining_epoch=pretraining_epoch;
		
		//System.out.println(this.layerSizes);
		for(int i=0;i<this.layerSizes.size();i++){
			if(i!=this.layerSizes.size()-1){
				SigmoidLayer sigmoidlayer=new SigmoidLayer(this.layerSizes.get(i),this.layerSizes.get(i+1));
				cRBM crbmlayer=new cRBM(this.layerSizes.get(i),this.layerSizes.get(i+1));
				crbmlayer.setK(K);
				crbmlayer.setAlpha(alpha);
				this.sigmoidlayers.put(i, sigmoidlayer);
				this.crbmlayers.put(i, crbmlayer);
			}else{
				OutputLayer outputlayer=new OutputLayer(this.layerSizes.get(i));
				this.outputlayer=outputlayer;
			}
		}
		
		
	}


	public void pretraining(RealMatrix X, int batch_size){
		

		// pretrain the RBMs
		Iterator it1=this.getCrbmlayers().entrySet().iterator();
		
		RealMatrix inputVs=null;
		
		while(it1.hasNext()){
			
			Entry entry=(Entry) it1.next();
			cRBM crbm=(cRBM) entry.getValue();
			int index=(int) entry.getKey();

			
			System.out.println("RBM layer "+index+":");
			
			for(int t=0;t<this.pretraining_epoch;t++){
				
				Random randomGenerator = new Random();

				if(index==0){
				
					RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);
					inputVs=inputData;
				
				}else{
					
					RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);	
					inputVs=this.forwardSample(inputData, index);
					System.out.println(inputVs.getRowDimension()+" "+inputVs.getColumnDimension());
				}
				
				System.out.println("pretraining_epoch:"+t+" ");
				
				
				crbm.train(inputVs, 1);
	
			}
			
		}
		
	}
	
	
	// sample output from initial RBM
	public RealMatrix forwardSample(RealMatrix inputData, int index){
		Iterator it1=this.getCrbmlayers().entrySet().iterator();
		
		RealMatrix inputVs=null;
		RealMatrix outputHs=null;
		while(it1.hasNext()){
			
			Entry entry=(Entry) it1.next();
			cRBM crbm=(cRBM) entry.getValue();
			int i=(int) entry.getKey();
			if(i == 0){
				inputVs=inputData;
			}else{
				inputVs=outputHs;
			}
			if(i==index){
				return inputVs;
			}
			outputHs=crbm.sampleHsGivenVs(inputVs, crbm.getRangeHidden());
			//System.out.println("hs:"+outputHs.getRowDimension()+" "+outputHs.getColumnDimension());
		}
		return null;
	}
	
	public void generateLayerSizes(int inputSize,int[] hiddenSizes,int outputSize){
		
		List<Integer> layerSizes=new ArrayList<Integer>();
		layerSizes.add(inputSize);
		for(int i=0;i<hiddenSizes.length;i++){
			layerSizes.add(hiddenSizes[i]);
		}
		layerSizes.add(outputSize);
		
		this.setLayerSizes(layerSizes);
	}

	public Map<Integer, SigmoidLayer> getSigmoidlayers() {
		return sigmoidlayers;
	}

	public void setSigmoidlayers(Map<Integer, SigmoidLayer> sigmoidlayers) {
		this.sigmoidlayers = sigmoidlayers;
	}


	public Map<Integer, cRBM> getCrbmlayers() {
		return crbmlayers;
	}

	public void setCrbmlayers(Map<Integer, cRBM> crbmlayers) {
		this.crbmlayers = crbmlayers;
	}

	public List<Integer> getLayerSizes() {
		return layerSizes;
	}

	public void setLayerSizes(List<Integer> layerSizes) {
		this.layerSizes = layerSizes;
	}

	public Map<Integer, String> getLayerTypes() {
		return layerTypes;
	}

	public void setLayerTypes(Map<Integer, String> layerTypes) {
		this.layerTypes = layerTypes;
	}

	public OutputLayer getOutputlayer() {
		return outputlayer;
	}

	public void setOutputlayer(OutputLayer outputlayer) {
		this.outputlayer = outputlayer;
	}

	public Map<Integer, RealMatrix> getWeights() {
		return weights;
	}

	public void setWeights(Map<Integer, RealMatrix> weights) {
		this.weights = weights;
	}

	public Map<Integer, RealMatrix> getBias() {
		return bias;
	}

	public void setBias(Map<Integer, RealMatrix> bias) {
		this.bias = bias;
	}

	public int getPretraining_epoch() {
		return pretraining_epoch;
	}

	public void setPretraining_epoch(int pretraining_epoch) {
		this.pretraining_epoch = pretraining_epoch;
	}
}
