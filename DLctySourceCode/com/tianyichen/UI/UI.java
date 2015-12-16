package com.tianyichen.UI;

import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.DBN.DBN;
import com.tianyichen.MLP.MLP;
import com.tianyichen.RBM.RBM;
import com.tianyichen.dao.DataStream;
import com.tianyichen.utils.otherTools;
import com.tianyichen.utils.printTool;

public class UI {
	
	public static void main(String[] args){
		
		
		
		
		String inputX="german_numer01_X.out";
		String inputY="german_numer01_Y.out";
		
		//String inputX="ijcnn1_X.out";
		//String inputY="ijcnn1_Y.out";
		

//		String inputX="GSE4226_4227_X.csv";
//		String inputY="GSE4226_4227_Y.csv";
		String inputtestX="GSE4229_X.csv";
		String inputtestY="GSE4229_Y.csv";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		
		Map<String, RealMatrix> infoTest=DataStream.loadData(inputtestX, inputtestY);
		RealMatrix testX=infoTest.get("X");
		RealMatrix testY=infoTest.get("Y");
//		
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		num_samples=Y.getRowDimension();
		
		System.out.println("input dataset size:");
		System.out.println("number of features: "+num_features+", number of samples:"+num_samples);
		
		/****** test RBM**********/
		
		
//		RBM rbm1=new RBM(num_features, num_features/2);
//		
//		System.out.println("W size:");
//		System.out.println("number of rows: "+rbm1.getW().getRowDimension()+", number of columns:"+rbm1.getW().getColumnDimension());
//		
//		System.out.println("hbias size:");
//		System.out.println("number of rows: "+rbm1.getHbias().getRowDimension()+", number of columns:"+rbm1.getHbias().getColumnDimension());
//
//		System.out.println("vbias size:");
//		System.out.println("number of rows: "+rbm1.getVbias().getRowDimension()+", number of columns:"+rbm1.getVbias().getColumnDimension());


//		int K=50;
//		double alpha=0.1;
//		rbm1.setK(K);
//		rbm1.setAlpha(alpha);
//		int training_epoches=100;
//		int batch_size=111;
//		
//		// train RBM
//		for(int t=0;t<training_epoches;t++){
//			Random randomGenerator = new Random();
//			
//			// select a batch of data randomly
//			RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);
//			
//			// update parameters of rbm1 with SGD
//			rbm1.updateParams(inputData);
//			//rbm1.getVbias();
//			double loss=rbm1.CrossEntropy(inputData);
//			System.out.println("training epoch:"+t+" loss value:"+loss);
//			
//		}
//		
		//printTool.matrix(rbm1.getW().getData());
		

		/****** test DBN**********/
		int training_epoches=10;
		int batch_size=111;
		int K=1;
		int pre_training_epoches=10;
		int[] hiddensizes={num_features/2,num_features*2,100};
		//int[] hiddensizes={num_features/2};
		double alpha=0.1;
		DBN dbn=new DBN(num_features,hiddensizes,2,K,pre_training_epoches,alpha);
		// pretraining RBMs.
		dbn.pretraining(X, batch_size);
	
		/***********test MLP **************/
		
//		int training_epoches=1000;
//		int pretrainingepoch=100;
//		int batch_size=23;
//		//int[] hiddensizes={num_features/2,num_features*2};
//		double lr=0.3; // learning rate of MLP
//		double alpha=0.1; // learning rate of DBN pretraining
//		int K=10;
//
//		int[] hiddensizes={140,100,100}; // hidden layer sizes.
//		MLP mlp=new MLP(X,num_features,hiddensizes,2,false,lr,batch_size,pretrainingepoch,alpha,K);
//		mlp.train(X, Y, batch_size, training_epoches);
//		
//		System.out.println("=================");
//		System.out.println("start to test:");
//		
//		RealMatrix test_Y=mlp.convertLabelToVector(testY,2);
//		mlp.predict(testX, test_Y);

		
		
	
	}	

}
