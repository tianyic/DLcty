package com.tianyichen.UI;

import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.DBN.DBN;
import com.tianyichen.RBM.RBM;
import com.tianyichen.cDBN.cDBN;
import com.tianyichen.dao.DataStream;

public class UIcDBN {
	
	public static void main(String[] argvs){
		
		String inputX="ijcnn1_X.out";
		String inputY="ijcnn1_Y.out";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		
		int batch_size=111;
		int K=1;
		int pre_training_epoches=10;
		int[] hiddensizes={10,100,50};
		//int[] hiddensizes={num_features/2};
		double alpha=0.1;
		cDBN cdbn=new cDBN(num_features,hiddensizes,2,K,pre_training_epoches,alpha);
		// pretraining RBMs.
		cdbn.pretraining(X, batch_size);
		//RBM rbm1=dbn.getRbmlayers().get(1);
	}

}
