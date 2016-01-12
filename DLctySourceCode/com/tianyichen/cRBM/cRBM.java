package com.tianyichen.cRBM;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.MultidimensionalCounter.Iterator;

import com.tianyichen.utils.Matrix;
import com.tianyichen.utils.Scalar;
import com.tianyichen.utils.otherTools;
import com.tianyichen.utils.printTool;

public class cRBM {
	
	
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
	
	// range of hidden units
	private List<Double> Dhidden;  
	
	//range of visible units
	private List<Double> DVisible;
	
	private Map rangeVisible;
	
	private Map rangeHidden;
	
	
	public cRBM(){}
	
	
	public cRBM(int num_visible,int num_hidden){
		
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
	
	
	// generate Range 	
	public Map generateRanges(RealMatrix samples){
		
		Map ranges=new HashMap();
		
		int num_samples=samples.getRowDimension();
		int num_units=samples.getColumnDimension();
		//System.out.println(num_samples+" "+num_units);
		
		double[] mean_units=Matrix.colMeans(samples);
		double[] var_units=Matrix.colVariances(samples);
		
		// construct normal distribution
		for(int i=0;i<num_units;i++){
			List range=new ArrayList();
			NormalDistribution norm=new NormalDistribution(mean_units[i],Math.sqrt(var_units[i])+0.01);
			for(int j=0;j<30;j++){
				double sample=norm.sample();
				range.add(sample);
			}
			ranges.put(i, range);
		}
		return ranges;
	}
	

	
	
	public Map probVisible(Map ranges_visible,RealMatrix hsample){
		
		Map probMap=new HashMap();
		
		int num_visible=this.n_visable;
			
		RealMatrix Wh=this.W.multiply(hsample);
		
		for(int i=0;i<num_visible;i++){
			
			List probs=new ArrayList();
			
			List range_visible=(List) ranges_visible.get(i);
			double denominator=0.0;
			double[] numerator=new double[range_visible.size()];
			double bi=this.vbias.getEntry(i, 0);
			for(int j=0;j<range_visible.size();j++){
				double vj=(double) range_visible.get(j);
				numerator[j]=Math.exp(bi*vj+vj*Wh.getEntry(i, 0));
				denominator+=numerator[j];				
			}
			
			for(int j=0;j<range_visible.size();j++){
				probs.add(numerator[j]/denominator);
			}
			
			probMap.put(i,probs);
			
		}
				
		
		return probMap;
		
	}
	
	public Map probHidden(Map ranges_hidden,RealMatrix vsample){
		
		Map probMap=new HashMap();
		
		int num_hidden=this.n_hidden;
			
		RealMatrix vTW=vsample.transpose().multiply(this.W);
		
		for(int i=0;i<num_hidden;i++){
			List probs=new ArrayList();
			
			List range_hidden=(List) ranges_hidden.get(i);
			double denominator=0.0;
			double[] numerator=new double[range_hidden.size()];
			double ci=this.hbias.getEntry(i, 0);
			for(int j=0;j<range_hidden.size();j++){
				double hj=(double) range_hidden.get(j);
				numerator[j]=Math.exp(ci*hj+hj*vTW.getEntry(0, i));
				denominator+=numerator[j];				
			}
			for(int j=0;j<range_hidden.size();j++){
				probs.add(numerator[j]/denominator);
			}
			
			probMap.put(i,probs);
			
		}				
		
		return probMap;
		
	}

	public RealMatrix sampleHsGivenVs(RealMatrix vsamples,Map ranges_hidden){
		
		int num_samples=vsamples.getRowDimension();
		
		Random rand=new Random();
		RealMatrix hsamples=MatrixUtils.createRealMatrix(num_samples, this.n_hidden);
		
		for(int i=0;i<num_samples;i++){
			Map probhidden=this.probHidden(ranges_hidden, vsamples.getRowMatrix(i).transpose());
			//System.out.println(probhidden);
			double[] hsample=new double[this.n_hidden];
			for(int l=0;l<this.n_hidden;l++){
				
				int index=0;
				double prob=rand.nextDouble();
				List probList=(List) probhidden.get(l);
				double sumprob=0.0;
				for(int j=0;j<probList.size();j++){
					if(sumprob<prob){
						index=j;
						break;
					}else{
						sumprob+=(double)probList.get(j);
						continue;
					}
				}
				hsample[l]=(double) ((List)ranges_hidden.get(l)).get(index);
				
				
			}
			hsamples.setRow(i, hsample);			
		}
		
		return hsamples;
		
	}
	
	public RealMatrix sampleVsGivenHs(RealMatrix hsamples,Map ranges_visible){
		
		int num_samples=hsamples.getRowDimension();
		Random rand=new Random();
		RealMatrix vsamples=MatrixUtils.createRealMatrix(num_samples, this.n_visable);
		
		for(int i=0;i<num_samples;i++){

			Map probvisible=this.probVisible(ranges_visible, hsamples.getRowMatrix(i).transpose());
			
			double[] vsample=new double[this.n_visable];
			
			for(int l=0;l<this.n_visable;l++){
				
				int index=0;
				double prob=rand.nextDouble();
				List probList=(List) probvisible.get(l);
				double sumprob=0.0;
				for(int j=0;j<probList.size();j++){
					if(sumprob<prob){
						index=j;
						break;
					}else{
						sumprob+=(double)probList.get(j);
						continue;
					}
				}
				vsample[l]=(double) ((List)ranges_visible.get(l)).get(index);
				
				
			}
			vsamples.setRow(i, vsample);			
		}
		
		return vsamples;
	}
	
	public RealMatrix GibbsVHV(RealMatrix vsamples,Map ranges_hidden,Map ranges_visible ){
		
		RealMatrix hsamples=this.sampleHsGivenVs(vsamples,ranges_hidden);
		RealMatrix vhvsamples=this.sampleVsGivenHs(hsamples, ranges_visible);
		return vhvsamples;
	}
	
	public RealMatrix initializeHsamples(Map ranges_hidden,int num_samples){
		Random rand = new Random();
		double[] hsample=new double[this.n_hidden];
		RealMatrix hsamples=MatrixUtils.createRealMatrix(num_samples, this.n_hidden);
		for(int l=0;l<num_samples;l++){
			for(int i=0;i<this.n_hidden;i++){
				List range=(List) ranges_hidden.get(i);
				
				// to do, generate an index randomly
				// then set the sampled one into hsamples
				int randomNum = rand.nextInt(range.size());
				hsample[i]=(double) range.get(randomNum);
			}
			hsamples.setRow(l, hsample);
		}
		
		return hsamples;
	}
	
	public Map initializeDH(Map ranges_visible){
		
		Map ranges_hidden=new HashMap();
		Random rand=new Random();
		for(int i=0;i<this.n_hidden;i++){
			int index=rand.nextInt(this.n_visable);
			List range=(List) ranges_visible.get(index);
			ranges_hidden.put(i, range);
			
			//System.out.println(index+" "+range);
		}
		
		return ranges_hidden;
	}
	
	public RealMatrix ContrasiveDivergence(RealMatrix vsamples,int K,Map ranges_visible,Map ranges_hidden){
		
		RealMatrix vsamples_old=vsamples;
		RealMatrix vsamples_new=null;
		for(int k=0;k<K;k++){
			vsamples_new=this.GibbsVHV(vsamples_old,ranges_hidden,ranges_visible);
			vsamples_old=vsamples_new;
		}
		return vsamples_new;
	}
	
	public void train(RealMatrix inputData,int training_epoches){

		
		Map ranges_visible=null;
		Map ranges_hidden=null;
		RealMatrix hsamples=null;
		RealMatrix vsamples=null;
		
		for(int t=0;t<training_epoches;t++){
			
			//System.out.println("Input data:"+inputData.getRowDimension()+" "+inputData.getColumnDimension());
			int num_samples=inputData.getRowDimension();
			
			// generate the range of visible units
			ranges_visible=this.generateRanges(inputData);
			//System.out.println(ranges_visible);
			this.rangeVisible=ranges_visible;						
			if(t==0){	
				ranges_hidden=this.initializeDH(ranges_visible);
				//System.out.println(ranges_hidden);
				this.rangeHidden=ranges_hidden;
				hsamples=this.initializeHsamples(ranges_hidden,num_samples);
			}else{
				ranges_hidden=this.generateRanges(hsamples);
				this.rangeHidden=ranges_hidden;
				hsamples=this.sampleHsGivenVs(inputData, ranges_hidden);
			}
			
			RealMatrix tildeVs=this.ContrasiveDivergence(inputData, 1, ranges_visible, ranges_hidden);
			
			this.updateParams(tildeVs,inputData,ranges_hidden);
			
		}
		
	}
	
	public void rbmtrain(RealMatrix X,int batch_size,int training_epoches){

		
		Map ranges_visible=null;
		Map ranges_hidden=null;
		RealMatrix hsamples=null;
		RealMatrix vsamples=null;
		
		for(int t=0;t<training_epoches;t++){
			Random randomGenerator = new Random();
			

			// select a batch of data randomly
			RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);
			
			//System.out.println("Input data:"+inputData.getRowDimension()+" "+inputData.getColumnDimension());
			int num_samples=inputData.getRowDimension();
			
			// generate the range of visible units
			ranges_visible=this.generateRanges(inputData);
			this.rangeVisible=ranges_visible;						
			if(t==0){	
				ranges_hidden=this.initializeDH(ranges_visible);
				this.rangeHidden=ranges_hidden;
				hsamples=this.initializeHsamples(ranges_hidden,num_samples);
			}else{
				ranges_hidden=this.generateRanges(hsamples);
				this.rangeHidden=ranges_hidden;
				hsamples=this.sampleHsGivenVs(inputData, ranges_hidden);
			}
			
			RealMatrix tildeVs=this.ContrasiveDivergence(inputData, 1, ranges_visible, ranges_hidden);
			
			this.updateParams(tildeVs,inputData,ranges_hidden);
			
		}
	}
		

	public void updateParams(RealMatrix tildeVs,RealMatrix inputData,Map ranges_hidden){
		
		RealMatrix tildeHs=this.sampleHsGivenVs(tildeVs,ranges_hidden);
		
		double inversenprime=1.0/((double)inputData.getRowDimension());
		
		RealMatrix hsamples=this.sampleHsGivenVs(inputData,ranges_hidden);
		//System.out.println(hsamples.getColumnDimension()+" "+hsamples.getRowDimension());
		//System.out.println(inputData.getColumnDimension()+" "+inputData.getRowDimension());
		RealMatrix pTermW=inputData.transpose().multiply(hsamples).scalarMultiply(-inversenprime);
		//printTool.matrix(hsamples.getData());
		//System.out.println(hsamples.getRowDimension()+" "+hsamples.getColumnDimension());
		RealMatrix pTermHbias=Matrix.mean(hsamples, 2).scalarMultiply(-inversenprime);
		RealMatrix pTermVbias=Matrix.mean(inputData, 2).scalarMultiply(-inversenprime);
		
		// negative terms for W, hbias, vbias
		RealMatrix nTermW=tildeVs.transpose().multiply(tildeHs).scalarMultiply(-inversenprime);
		RealMatrix nTermHbias=Matrix.mean(tildeHs, 2).scalarMultiply(-inversenprime);
		RealMatrix nTermVbias=Matrix.mean(tildeVs, 2).scalarMultiply(-inversenprime);

		RealMatrix gradW=pTermW.add(nTermW);
		RealMatrix gradHbias=pTermHbias.add(nTermHbias);
		RealMatrix gradVbias=pTermVbias.add(nTermVbias);
		
		this.setW(this.getW().add(gradW.scalarMultiply(this.alpha)));
		this.setHbias(this.getHbias().add(gradHbias.scalarMultiply(this.getAlpha())));
		this.setVbias(this.getVbias().add(gradVbias.scalarMultiply(this.getAlpha())));

		
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


	public List<Double> getDhidden() {
		return Dhidden;
	}


	public void setDhidden(List<Double> dhidden) {
		Dhidden = dhidden;
	}


	public List<Double> getDVisible() {
		return DVisible;
	}


	public void setDVisible(List<Double> dVisible) {
		DVisible = dVisible;
	}


	public Map getRangeVisible() {
		return rangeVisible;
	}


	public void setRangeVisible(Map rangeVisible) {
		this.rangeVisible = rangeVisible;
	}


	public Map getRangeHidden() {
		return rangeHidden;
	}


	public void setRangeHidden(Map rangeHidden) {
		this.rangeHidden = rangeHidden;
	}
	
	
}
