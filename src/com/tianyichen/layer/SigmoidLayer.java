package com.tianyichen.layer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.utils.Matrix;
import com.tianyichen.utils.Scalar;

public class SigmoidLayer {
	
	private int n_input;
	private int n_output;
	private RealMatrix weights;
	private RealMatrix bias;
	private RealMatrix pre_sigmoid_term;
	
	public SigmoidLayer(int n_input,int n_output){
		this.n_input=n_input;
		this.n_output=n_output;
		
	}
	
	public void initializeBW(){
		
		NormalDistribution nd=new NormalDistribution();
		double[] bias=new double[this.n_output];
		for(int i=0;i<this.n_output;i++){
			bias[i]=nd.sample();
		}
		this.setBias(MatrixUtils.createColumnRealMatrix(bias));
		
		this.weights=MatrixUtils.createRealMatrix(n_input, n_output);
		for(int i=0;i<this.n_input;i++){			
			for(int j=0;j<this.n_output;j++){
				this.weights.setEntry(i, j, nd.sample());
			}
		}
		
	}
	
	public RealMatrix sigmoidOutput(RealMatrix input){
		// assume size of input is number_features by num_samples
		RealMatrix wx_b=Matrix.addV(this.weights.transpose().multiply(input), this.bias);
		this.pre_sigmoid_term=wx_b;
		return Scalar.sigmoid(wx_b);
	}

	public int getN_input() {
		return n_input;
	}


	public void setN_input(int n_input) {
		this.n_input = n_input;
	}


	public int getN_output() {
		return n_output;
	}


	public void setN_output(int n_output) {
		this.n_output = n_output;
	}


	public RealMatrix getWeights() {
		return weights;
	}


	public void setWeights(RealMatrix weights) {
		this.weights = weights;
	}


	public RealMatrix getBias() {
		return bias;
	}


	public void setBias(RealMatrix bias) {
		this.bias = bias;
	}

	public RealMatrix getPre_sigmoid_term() {
		return pre_sigmoid_term;
	}

	public void setPre_sigmoid_term(RealMatrix pre_sigmoid_term) {
		this.pre_sigmoid_term = pre_sigmoid_term;
	}

	
}
