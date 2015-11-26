package com.tianyichen.RBM;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.utils.Matrix;
import com.tianyichen.utils.Scalar;
import com.tianyichen.utils.printTool;

public class RBM {
	
	
	// weights
	private RealMatrix W;
	
	// bias of hidden layer
	private RealMatrix hbias;
	
	// bias of visible layer
	private RealMatrix vbias;
	
	// 
	private int n_visable;
	
	private int n_hidden;
	
	// CD-K's K
	private int K;
	
	private double alpha;
	
	public RBM(){}
	
	
	public RBM(int num_visible,int num_hidden){
		
		this.W=MatrixUtils.createRealMatrix(num_visible, num_hidden);
		this.n_hidden=num_hidden;
		this.n_visable=num_visible;
		
		
		// initiliaze W by a uniform distribution
		double lower=-4*Math.sqrt(6.0/((double)this.n_hidden+(double)this.n_visable));
		double upper=-lower;
		
		UniformRealDistribution uniform=new UniformRealDistribution(lower,upper);
		
		for(int i=0;i<this.W.getRowDimension();i++){
			for(int j=0;j<this.W.getColumnDimension();j++){
				this.W.setEntry(i, j, uniform.sample());
			}
		}
		
		double[] hbiasprior=new double[this.n_hidden];
		double[] vbiasprior=new double[this.n_visable];
		
		for(int i=0;i<this.n_hidden;i++){
			hbiasprior[i]=0.0;
		}
		this.hbias=MatrixUtils.createColumnRealMatrix(hbiasprior);
		
		for(int i=0;i<this.n_visable;i++){
			vbiasprior[i]=0.0;
		}
		this.vbias=MatrixUtils.createColumnRealMatrix(vbiasprior);
	}
	
	// forward visible layer to hidden layer
	public RealMatrix forwardProp(RealMatrix v_sample){
		RealMatrix pre_sigmoid_activation=this.getW().transpose().multiply(v_sample).add(this.hbias);
		RealMatrix sigmoidterm=Scalar.sigmoid(pre_sigmoid_activation);
		return sigmoidterm;
	}
	
	// backward hidden layer to visible layer
	public RealMatrix backwardProp(RealMatrix h_sample){
		
		RealMatrix pre_sigmoid_activation=this.getW().multiply(h_sample).add(this.vbias);
		RealMatrix sigmoidterm=Scalar.sigmoid(pre_sigmoid_activation);
		return sigmoidterm;
	}

	public RealMatrix sampleVgivenH(RealMatrix h_sample){
		
		RealMatrix p_vector=this.backwardProp(h_sample);
		
		RealMatrix v1sample=Scalar.sampleBio(p_vector);
		
		return v1sample;
	}
	
	public RealMatrix sampleHgivenV(RealMatrix v_sample){
		
		RealMatrix p_vector=this.forwardProp(v_sample);
		
		RealMatrix h1sample=Scalar.sampleBio(p_vector);
		
		return h1sample;
	}
	
	public Map<String, RealMatrix> gibbsVHV(RealMatrix v_sample){
		
		RealMatrix h1_sample=this.sampleHgivenV(v_sample);
		RealMatrix v1_sample=this.sampleVgivenH(h1_sample);
		Map<String, RealMatrix> infoMap=new HashMap<String, RealMatrix>();
		infoMap.put("h", h1_sample);
		infoMap.put("v", v1_sample);
		return infoMap;
	}
	
	public Map<String, RealMatrix> gibbsHVH(RealMatrix h_sample){
		RealMatrix v1_sample=this.sampleVgivenH(h_sample);
		RealMatrix h1_sample=this.sampleHgivenV(v1_sample);
		Map<String, RealMatrix> infoMap=new HashMap<String, RealMatrix>();
		infoMap.put("h", h1_sample);
		infoMap.put("v", v1_sample);		
		return infoMap;
	}
	
	public RealMatrix ContrasiveDivergence(RealMatrix v_sample){
		RealMatrix tilde_v=MatrixUtils.createRealMatrix(v_sample.getRowDimension(), v_sample.getColumnDimension());
		
		RealMatrix new_v=null;
		RealMatrix old_v=v_sample;
		for(int k=0;k<this.K;k++){
			if(k>0){
				old_v=new_v;
			}
			Map<String, RealMatrix> infoMap=this.gibbsVHV(old_v);
			new_v=infoMap.get("v");
		}
		tilde_v=new_v;
		return tilde_v;
	}
	
	
	public RealMatrix sampleHsgivenVs(RealMatrix v_samples){
		//System.out.println(this.n_hidden);
		RealMatrix Hs=MatrixUtils.createRealMatrix(this.n_hidden, v_samples.getRowDimension());
		int num_samples=v_samples.getRowDimension();
		
		for(int i=0;i<num_samples;i++){
			RealMatrix h1sample=this.sampleHgivenV(v_samples.transpose().getColumnMatrix(i));
			Hs.setColumn(i, h1sample.getColumn(0));
		}
		
		return Hs;
	}
	
	public RealMatrix CD_parallel(RealMatrix v_samples){
		// assume all data has the size number_samples by number_features
		
		// tilde_V size is number_samples by number_features since all samples assume v_sample, h_sample are column matrix
		RealMatrix tilde_Vs=MatrixUtils.createRealMatrix(v_samples.getColumnDimension(), v_samples.getRowDimension());
		int num_samples=v_samples.getRowDimension();
		
		for(int i=0;i<num_samples;i++){
			
			RealMatrix tilde_v=this.ContrasiveDivergence(v_samples.transpose().getColumnMatrix(i));
			//System.out.println(tilde_v.getRowDimension()+" "+tilde_v.getColumnDimension());
			tilde_Vs.setColumn(i, tilde_v.getColumn(0));
			
		}
		
		return tilde_Vs;
	}
	
	public RealMatrix reconstructV(RealMatrix inputData){
		
		int num_features=inputData.getColumnDimension();
		int num_samples=inputData.getRowDimension();
		
		//System.out.println(num_samples+" "+num_features);
		RealMatrix new_Vsigmoids=MatrixUtils.createRealMatrix(inputData.getColumnDimension(), inputData.getRowDimension());
		for(int i=0;i<num_samples;i++){
			RealMatrix real_v=inputData.transpose().getColumnMatrix(i);
			Map infoMap=this.gibbsVHV(real_v);
			RealMatrix new_v=(RealMatrix) infoMap.get("v");
			RealMatrix new_h=(RealMatrix) infoMap.get("h");
			RealMatrix pre_sigmoid_nv=this.backwardProp(new_h);
			//printTool.matrix(this.backwardProp(new_h).getData());
			new_Vsigmoids.setColumn(i, pre_sigmoid_nv.getColumn(0));
		}
		return new_Vsigmoids;
	}
	
	
	public void updateParams(RealMatrix inputData){
		
		// sample tilde Vs
		RealMatrix tilde_Vs=this.CD_parallel(inputData);
		
		//printTool.matrix(tilde_Vs.transpose().getData());
		
		// sample h
		RealMatrix Hvs=Scalar.sigmoid(Matrix.addV(this.W.transpose().multiply(inputData.transpose()), this.hbias));
		//printTool.matrix(Hvs.transpose().getData());
		//RealMatrix Hs=this.sampleHsgivenVs(inputData);
		//printTool.matrix(Hs.transpose().getData());
		
		// sample tilde_Hs
		RealMatrix tilde_Hvs=Scalar.sigmoid(Matrix.addV(this.W.transpose().multiply(tilde_Vs), this.hbias));
		//RealMatrix tilde_Hs=this.sampleHsgivenVs(tilde_Vs.transpose());
		//printTool.matrix(tilde_Hs.transpose().getData());
		
		//System.out.println("number of rows: "+inputData.getRowDimension()+", number of columns:"+inputData.getColumnDimension());
		//System.out.println("number of rows: "+tilde_Vs.getRowDimension()+", number of columns:"+tilde_Vs.getColumnDimension());
		//System.out.println("number of rows: "+Hs.getRowDimension()+", number of columns:"+Hs.getColumnDimension());

		double inversenprime=1.0/((double)inputData.getRowDimension());
		//System.out.println(inversenprime);
		// postive terms for W, hbias, vbias
		
		RealMatrix pTermW=Hvs.multiply(inputData).transpose().scalarMultiply(inversenprime);
		RealMatrix pTermHbias=Matrix.mean(Hvs, 1).scalarMultiply(inversenprime);
		RealMatrix pTermVbias=Matrix.mean(inputData.transpose(), 1).scalarMultiply(inversenprime);
		
		// negative terms for W, hbias, vbias
		RealMatrix nTermW=tilde_Vs.multiply(tilde_Hvs.transpose()).scalarMultiply(-inversenprime);
		RealMatrix nTermHbias=Matrix.mean(tilde_Hvs, 1).scalarMultiply(-inversenprime);
		RealMatrix nTermVbias=Matrix.mean(tilde_Vs, 1).scalarMultiply(-inversenprime);
		
		// gradients for W, hbias, vbias
		
		RealMatrix gradW=pTermW.add(nTermW);
		RealMatrix gradHbias=pTermHbias.add(nTermHbias);
		RealMatrix gradVbias=pTermVbias.add(nTermVbias);
		
		this.setW(this.getW().add(gradW.scalarMultiply(this.alpha)));
		this.setHbias(this.getHbias().add(gradHbias.scalarMultiply(this.getAlpha())));
		this.setVbias(this.getVbias().add(gradVbias.scalarMultiply(this.getAlpha())));
		
		
	}
	
	public double CrossEntropy(RealMatrix inputData){
		
		RealMatrix new_Vsigmoids=this.reconstructV(inputData);
		//printTool.matrix(new_Vsigmoids.getData());
		RealMatrix old_Vs=inputData.transpose();
		
		RealMatrix crossLossM=Scalar.crossLoss(new_Vsigmoids, old_Vs);
		
		double crossEntropyLoss=Matrix.meanMatrix(crossLossM);
		return crossEntropyLoss;
		
	}
	
	public RealMatrix freeEnergy(RealMatrix vm_sample){
		// dimension of vm_sample is features by batch size
		// dimension of W is visible features by hidden features
		RealMatrix wx=vm_sample.transpose().multiply(this.W);
		return null;
	}

	public RealMatrix getW() {
		return W;
	}


	public void setW(RealMatrix w) {
		W = w;
	}


	public RealMatrix getHbias() {
		return hbias;
	}


	public void setHbias(RealMatrix hbias) {
		this.hbias = hbias;
	}


	public RealMatrix getVbias() {
		return vbias;
	}


	public void setVbias(RealMatrix vbias) {
		this.vbias = vbias;
	}


	public int getN_visable() {
		return n_visable;
	}


	public void setN_visable(int n_visable) {
		this.n_visable = n_visable;
	}


	public int getN_hidden() {
		return n_hidden;
	}


	public void setN_hidden(int n_hidden) {
		this.n_hidden = n_hidden;
	}


	public int getK() {
		return K;
	}


	public void setK(int k) {
		K = k;
	}


	public double getAlpha() {
		return alpha;
	}


	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	
}
