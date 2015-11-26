package com.tianyichen.dao;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.utils.TypeConvert;

public class DataStream {
	
	public static Map<String, RealMatrix> loadData(String inputX, String inputY){
		
		File fileX=new File(inputX);
		File fileY=new File(inputY);
		
		BufferedReader xreader=null;
		BufferedReader yreader=null;
		
		Map<String, RealMatrix> infoMap=new HashMap();
		try {
			xreader=new BufferedReader(new FileReader(fileX));
			yreader=new BufferedReader(new FileReader(fileY));
			
			List<List> X=new ArrayList<List>();
			List Y=new ArrayList();
			
			String lineX=null;
			String lineY=null;
			int count=0;
			while((lineX=xreader.readLine())!=null && (lineY=yreader.readLine())!=null){
				
				List x=new ArrayList();

				String[] linesplit=lineX.split(",");
				
				for(int i=0;i<linesplit.length;i++){
					x.add(Double.parseDouble(linesplit[i]));	
				}
				X.add(x);
				Y.add(Double.parseDouble(lineY));
				
			}
			
			RealMatrix mX=MatrixUtils.createRealMatrix(TypeConvert.ArrayTodoublem(X));
			RealMatrix mY=MatrixUtils.createColumnRealMatrix(TypeConvert.ArrayTodoublev(Y));
			infoMap.put("X", mX);
			infoMap.put("Y", mY);
			return infoMap;
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		return null;
	}
	 
	
}
