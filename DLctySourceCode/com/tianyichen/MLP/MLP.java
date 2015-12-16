package com.tianyichen.MLP;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.DBN.DBN;
import com.tianyichen.cost.CrossEntropyCost;
import com.tianyichen.utils.Matrix;
import com.tianyichen.utils.Scalar;
import com.tianyichen.utils.otherTools;
import com.tianyichen.utils.printTool;

public class MLP {
	
	private List<Integer> layerSizes;
	private Map<Integer, RealMatrix> weights;
	private Map<Integer, RealMatrix> bias;
	private double lr;
	private Map<Integer,RealMatrix> activations;
	private Map<Integer, RealMatrix> pre_sigmoid_activation;
	private Map<Integer,RealMatrix> delta;
	private Map<Integer, RealMatrix> grad_b;
	private Map<Integer, RealMatrix> grad_w;
	
	private int num_sigmoidlayers;
	
	public MLP(){}
	
	public MLP(RealMatrix X,int inputSize,int[] hiddenSizes,int outputSize, boolean needPretrain,double lr,int batch_size,int pre_training_epoches,double alpha, int K ){
		
		this.setLr(lr);
		this.bias=new HashMap<Integer,RealMatrix>();
		this.weights=new HashMap<Integer, RealMatrix>();
		this.grad_b=new HashMap<Integer, RealMatrix>();
		this.grad_w=new HashMap<Integer, RealMatrix>();
		this.delta=new HashMap<Integer, RealMatrix>();
		this.activations=new HashMap<Integer, RealMatrix>();
		this.pre_sigmoid_activation=new HashMap<Integer, RealMatrix>();
		
		this.generateLayerSizes(inputSize, hiddenSizes, outputSize);
		this.num_sigmoidlayers=this.layerSizes.size()-1;
		System.out.println("num_sigmoidlayers: "+this.num_sigmoidlayers);
		System.out.println(this.layerSizes);
		
		if(needPretrain==false){
			for(int i=0;i<this.num_sigmoidlayers;i++){
				RealMatrix weight=this.initializeWeights(this.layerSizes.get(i), this.layerSizes.get(i+1));
				RealMatrix bias=this.initializeBias(this.layerSizes.get(i+1));
				this.weights.put(i, weight);
				this.bias.put(i, bias);
			}
			//System.out.println(this.weights);
			//System.out.println(this.bias);
		}else if(needPretrain==true){
			DBN dbn=new DBN(inputSize,hiddenSizes,outputSize,K,pre_training_epoches,alpha);
			dbn.pretraining(X, batch_size);
			
			for(int i=0;i<this.num_sigmoidlayers;i++){
				RealMatrix weight=dbn.getRbmlayers().get(i).getW();
				RealMatrix bias=dbn.getRbmlayers().get(i).getHbias();
				this.weights.put(i, weight);
				this.bias.put(i, bias);
			}
		}
	}

	public void feedforward(RealMatrix input){
		
		this.activations.put(-1, input.transpose());
		
		RealMatrix input_v=null;
		RealMatrix output_v=null;
		
		
		for(int i=0;i<this.num_sigmoidlayers;i++){
			if(i==0){
				input_v=input.transpose();
			}else{
				input_v=output_v;
			}
			
			RealMatrix weight=this.getWeights().get(i);
			RealMatrix bias=this.getBias().get(i);
			
			List<RealMatrix> list=this.sigmoidOutput(input_v, weight, bias);
			output_v=list.get(1);
			this.activations.put(i, output_v);
			this.pre_sigmoid_activation.put(i, list.get(0));
		}
		
		//System.out.println(this.activations.get(2).getRowDimension()+" "+this.activations.get(2).getColumnDimension());
		//System.out.println(this.pre_sigmoid_activation);
		
	}
	
	public RealMatrix predict(RealMatrix input, RealMatrix Y){
		
		//printTool.matrix(Y.getData());
		RealMatrix input_v=null;
		RealMatrix output_v=null;
		
		//printTool.matrix(input.getData());
		
		for(int i=0;i<this.num_sigmoidlayers;i++){
			if(i==0){
				input_v=input.transpose();
			}else{
				input_v=output_v;
			}
			
			
			RealMatrix weight=this.getWeights().get(i);
			RealMatrix bias=this.getBias().get(i);
			output_v=this.sigmoidOutput(input_v, weight, bias).get(1);
			//printTool.matrix(weight.getData());
		}
		//printTool.shape(output_v);
		//System.out.println(output_v.getEntry(0, 3));
		//System.out.println(output_v.getEntry(1, 3));
		//printTool.matrix(output_v.getData());
		//printTool.matrix(Y.getData());
		//System.out.println(Y.getEntry(2, 0));
		double correctCount=0.0;
		for(int i=0;i<output_v.getColumnDimension();i++){
			double[] activation_sample=output_v.getColumn(i);
			int maxindexX=otherTools.argmax(activation_sample);
			
			double[] labelY=Y.getColumn(i);
			int maxindexY=otherTools.argmax(labelY);
			if(maxindexY ==maxindexX){
				correctCount++;
			}
			//System.out.print(maxindex+" ");
		}
		double accuracy=correctCount/(double)output_v.getColumnDimension();
		System.out.print("predict accuracy is:"+accuracy);
		//System.out.println(accuracy);
		return output_v;
	}
	

	
	public void train(RealMatrix X, RealMatrix Y,int batch_size,int train_epoches){
		
		Random randomGenerator = new Random();
		for(int t=0;t<train_epoches;t++){
			System.out.print("training epoch:"+t+" validating data");
			List<RealMatrix> inputData=otherTools.stochasticSubmatrixLabel(X,Y, batch_size, randomGenerator);
			this.feedforward(inputData.get(0));
			
			RealMatrix real_Y=convertLabelToVector(inputData.get(1),2);
			
			this.backwardProp(real_Y);
			//RealMatrix predict_ac=this.predict(inputData.get(0),real_Y);
			//printTool.matrix(predict_ac.getData());
			this.updateParams();
			
			RealMatrix real_Y_all=convertLabelToVector(Y,2);
			//printTool.shape(real_Y_all);
			//printTool.shape(X);
			RealMatrix predict_ac1=this.predict(X,real_Y_all);
			//printTool.matrix(predict_ac1.getData());
			//System.out.println(1);
			System.out.println();
		}
	}
	
	public void backwardProp(RealMatrix Y){
		
		int lastindex=this.num_sigmoidlayers-1;
		RealMatrix lastdelta=CrossEntropyCost.calDelta(this.activations.get(lastindex), Y);
		//printTool.matrix(lastdelta.getData());
		//printTool.shape(lastdelta);
		
		double batch_size=(double)lastdelta.getColumnDimension();
		//System.out.println(batch_size);
		
		RealMatrix last_grad_b=Matrix.mean(lastdelta, 1);
		//printTool.matrix(last_grad_b.getData());
		
		RealMatrix last_grad_w=this.activations.get(lastindex-1).multiply(lastdelta.transpose()).scalarMultiply(1.0/(double)batch_size);
		//printTool.shape(last_grad_w);
		//printTool.matrix(last_grad_w.getData());
		
		this.grad_b.put(lastindex, last_grad_b);
		this.grad_w.put(lastindex, last_grad_w);
		this.delta.put(lastindex, lastdelta);
		
		for(int i=lastindex-1;i>=0;i--){
			
			RealMatrix sigmoidprime=Scalar.sigmoidprime(this.pre_sigmoid_activation.get(i));
			//printTool.matrix(this.pre_sigmoid_activation.get(i).getData());
			//printTool.matrix(sigmoidprime.getData());
			RealMatrix delta=Scalar.multiply(this.getWeights().get(i+1).multiply(this.delta.get(i+1)),sigmoidprime);
			
			RealMatrix grad_b=Matrix.mean(delta, 1);
			RealMatrix grad_w=this.activations.get(i-1).multiply(delta.transpose()).scalarMultiply(1.0/(double)batch_size);

			//printTool.matrix(delta.getData());
			//printTool.shape(delta);
//			printTool.matrix(grad_b.getData());
//			printTool.shape(grad_b);
//			printTool.matrix(grad_w.getData());
//			printTool.shape(grad_w);
			this.delta.put(i, delta);
			this.grad_b.put(i, grad_b);
			this.grad_w.put(i, grad_w);
			
		}
		
	}
	
	public void updateParams(){
		for(int i=this.num_sigmoidlayers-1;i>=0;i--){
			this.weights.put(i, this.weights.get(i).add(this.grad_w.get(i).scalarMultiply(-this.lr)));
			this.bias.put(i, this.bias.get(i).add(this.grad_b.get(i).scalarMultiply(-this.lr)));
		}
	}
	
	public RealMatrix initializeWeights(int n_input,int n_output){
		

		NormalDistribution nd=new NormalDistribution();
		RealMatrix weight=MatrixUtils.createRealMatrix(n_input, n_output);
		for(int i=0;i<n_input;i++){			
			for(int j=0;j<n_output;j++){
				weight.setEntry(i, j, nd.sample());
			}
		}
		return weight;
		
	}
	public RealMatrix initializeBias(int n_output){
		
		NormalDistribution nd=new NormalDistribution();
		double[] bias=new double[n_output];
		for(int i=0;i<n_output;i++){
			bias[i]=nd.sample();
		}
		return MatrixUtils.createColumnRealMatrix(bias);
		
	}
	
	public List<RealMatrix> sigmoidOutput(RealMatrix input,RealMatrix weight,RealMatrix bias){
		// assume size of input is number_features by num_samples
		RealMatrix wx_b=Matrix.addV(weight.transpose().multiply(input),bias);
		List<RealMatrix> list=new ArrayList<RealMatrix>();
		list.add(wx_b);
		list.add(Scalar.sigmoid(wx_b));
		
		// index 0 is sigmoid term, index 1 is activation
		return list;
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
	public static RealMatrix convertLabelToVector(RealMatrix Y,int vector_len){
		
		RealMatrix labelmatrix=MatrixUtils.createRealMatrix(vector_len, Y.getRowDimension());
		
		for(int i=0;i<Y.getRowDimension();i++){
			if(Y.getEntry(i, 0)==1){
				labelmatrix.setEntry(0, i, 1);
				labelmatrix.setEntry(1, i, 0);
			}else if(Y.getEntry(i, 0)==-1){
				labelmatrix.setEntry(0, i, 0);
				labelmatrix.setEntry(1, i, 1);
			}
		}
		
		return labelmatrix;
	}
	public List<Integer> getLayerSizes() {
		return layerSizes;
	}

	public void setLayerSizes(List<Integer> layerSizes) {
		this.layerSizes = layerSizes;
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

	public double getLr() {
		return lr;
	}

	public void setLr(double lr) {
		this.lr = lr;
	}

	public Map<Integer, RealMatrix> getActivations() {
		return activations;
	}

	public void setActivations(Map<Integer, RealMatrix> activations) {
		this.activations = activations;
	}

	public Map<Integer, RealMatrix> getPre_sigmoid_activation() {
		return pre_sigmoid_activation;
	}

	public void setPre_sigmoid_activation(Map<Integer, RealMatrix> pre_sigmoid_activation) {
		this.pre_sigmoid_activation = pre_sigmoid_activation;
	}

	public Map<Integer, RealMatrix> getGrad_b() {
		return grad_b;
	}

	public void setGrad_b(Map<Integer, RealMatrix> grad_b) {
		this.grad_b = grad_b;
	}

	public Map<Integer, RealMatrix> getGrad_w() {
		return grad_w;
	}

	public void setGrad_w(Map<Integer, RealMatrix> grad_w) {
		this.grad_w = grad_w;
	}

	public int getNum_sigmoidlayers() {
		return num_sigmoidlayers;
	}

	public void setNum_sigmoidlayers(int num_sigmoidlayers) {
		this.num_sigmoidlayers = num_sigmoidlayers;
	}
	
	

}
