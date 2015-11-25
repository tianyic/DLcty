package com.tianyichen.layer;

public class SigmoidLayer {
	
	private int n_input;
	private int n_output;
	
	
	public SigmoidLayer(int n_input,int n_output){
		this.n_input=n_input;
		this.n_output=n_output;
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

	
}
