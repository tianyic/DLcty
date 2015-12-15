import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.MLP.MLP;
import com.tianyichen.dao.DataStream;


public class testMLP {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String inputX="code/GSE4226_4227_X.csv";
		String inputY="code/GSE4226_4227_Y.csv";
		String inputtestX="code/GSE4229_X.csv";
		String inputtestY="code/GSE4229_Y.csv";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		
		Map<String, RealMatrix> infoTest=DataStream.loadData(inputtestX, inputtestY);
		RealMatrix testX=infoTest.get("X");
		RealMatrix testY=infoTest.get("Y");
		
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		int training_epoches=1000;
		int pretrainingepoch=10;
		int batch_size=23;
		
		//int[] hiddensizes={num_features/2,num_features*2};
		double lr=0.3; // learning rate of MLP
		double alpha=0.1; // learning rate of DBN pretraining
		int K=10;

		int[] hiddensizes={140,100,100}; // hidden layer sizes.
		int outputsize=2;
		MLP mlp=new MLP(X,num_features,hiddensizes,outputsize,false,lr,batch_size,pretrainingepoch,alpha,K);
		mlp.train(X, Y, batch_size, training_epoches);
		
		System.out.println("=================");
		System.out.println("start to predict labels of test dataset:");
		RealMatrix test_Y=mlp.convertLabelToVector(testY,2);
		mlp.predict(testX, test_Y);
	
		Map weightsMap=mlp.getWeights();
		Map biasMap=mlp.getBias();
		System.out.println();
		//RealMatrix weights=(RealMatrix) mlp.getWeights();
		System.out.print("MLP size:"+num_features+" ");
		for (int i=0;i<hiddensizes.length;i++ ) {
			System.out.print(hiddensizes[i]+" ");
			
		}
		System.out.print(outputsize);
		System.out.println();
	}

}
