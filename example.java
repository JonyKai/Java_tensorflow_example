package user;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import javax.imageio.ImageIO;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;

import sun.misc.IOUtils;
public class Example {
	 private static final String imageFile= "./test.jpg";
	 private static final String modelFile = "./frozen_model.pb";
	 private static final float dropoutValue = (float) 0.7;
	 private static final String[] numString = new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", ""};
	 public static void main(String[] args) {
		 try {
			BufferedImage imageBuffer = ImageIO.read(new File(imageFile));
			String result = recNum(imageBuffer);
		} catch (IOException e) {
			System.out.println("Can't find file!")
			e.printStackTrace();
		}
	 }

	 static public String recNum(BufferedImage input) {
		 /**
		  * main
		  */
		 byte[] imageBytes = null;
		 Path modelPath = Paths.get(modelFile);
		 byte[] graphDef = readAllBytesOrExit(modelPath);
		 // convert RGB image to gray
		 BufferedImage grayImage = new BufferedImage(input.getWidth(), input.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		 new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(input, grayImage);
		 ByteArrayOutputStream out = new ByteArrayOutputStream();
		 try {
			ImageIO.write(grayImage, "jpg", out);
		} catch (IOException e) {
			e.printStackTrace();
		}
		 imageBytes = out.toByteArray();
		 Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
		 Tensor<Boolean> is_training = constructAndExecuteGraphToIsTraining();
		 Tensor<Float> dropout = constructAndExecuteGraphToDropout();
		 long[][] digits_prediction = executeInceptionGraph(graphDef, image, is_training, dropout);
		 int length = digits_prediction[0].length;
		 String output = "";
		 for(int i=0;i<length;i++) {
	    	output += numString[(int) digits_prediction[0][i]];
   			System.out.println(digits_prediction[0][i]);
		 }
	    return output;
	 }
	 private static int colorToRGB(int alpha, int red, int green, int blue) {
		  int newPixel = 0;
		  newPixel += alpha;
		  newPixel = newPixel << 8;
		  newPixel += red;
		  newPixel = newPixel << 8;
		  newPixel += green;
		  newPixel = newPixel << 8;
		  newPixel += blue;
		  
		  return newPixel;
	 }
	 private static BufferedImage convertRGBToGray(BufferedImage newPic) {
		 BufferedImage grayImage = new BufferedImage(newPic.getWidth(), newPic.getHeight(),  BufferedImage.TYPE_USHORT_GRAY);   
		 for (int i = 0; i < newPic.getWidth(); i++) {
			 for (int j = 0; j < newPic.getHeight(); j++) {
				 final int color = newPic.getRGB(i, j);
				 final int r = (color >> 16) & 0xff;
				 final int g = (color >> 8) & 0xff;
				 final int b = color & 0xff;
				 int gray = (int) (0.3 * r + 0.59 * g + 0.11 * b);
				 int newPixel = colorToRGB(255, gray, gray, gray);
				 grayImage.setRGB(i, j, newPixel);
				  }
				 }
		 return grayImage;
	 }
	 private static byte[] rgbToGrayToByte(BufferedImage image) {
		 BufferedImage grayImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		 new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(image, grayImage);
		 return (byte[])grayImage.getData().getDataElements(0, 0, image.getWidth(), image.getHeight(), null);
	 }
	 
	 private static byte[] readAllBytesOrExit(Path path) {
		    try {
		      return Files.readAllBytes(path);
		    } catch (IOException e) {
		      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
		      System.exit(1);
		    }
		    return null;
		  }
	private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
	    try (Graph g = new Graph()) {
	    	GraphBuilder b = new GraphBuilder(g);
	    	final float scale = (float) 255.0;
	      // Since the graph is being constructed once per execution here, we can use a constant for the
	      // input image. If the graph were to be re-used for multiple input images, a placeholder would
	      // have been more appropriate.
	    	final Output<String> input = b.constant("input", imageBytes);
	    	final int H=40;
	    	final int W=120;
	    	final Output<Float> output = b.div(b.resizeBilinear(b.expandDims(
                  	b.cast(b.decodeJpeg(input, 0), Float.class),
                  	b.constant("make_batch", 0)),b.constant("size", new int[] {H, W})), b.constant("scale", scale));
	    	try (Session s = new Session(g)) {
	    		return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
	    	}
	    }
	 }
	private static Tensor<Boolean> constructAndExecuteGraphToIsTraining() {
	    try (Graph g = new Graph()) {
	    	GraphBuilder b = new GraphBuilder(g);
	    	final Output<Boolean> is_training = b.constant("is_training", false);
	    	try (Session s = new Session(g)) {
	    		return s.runner().fetch(is_training.op().name()).run().get(0).expect(Boolean.class);
	    	}
	    }
	 } 
	private static Tensor<Float>  constructAndExecuteGraphToDropout(){
		try (Graph g = new Graph()) {
	    	GraphBuilder b = new GraphBuilder(g);
	    	final Output<Float> dropout = b.constant("dropout", dropoutValue);
	    	try (Session s = new Session(g)) {
	    		return s.runner().fetch(dropout.op().name()).run().get(0).expect(Float.class);
	    	}
	    }
	}
	private static long[][] executeInceptionGraph(byte[] graphDef, Tensor<Float> image, Tensor<Boolean> is_training, Tensor<Float> dropout) {
	    try (Graph g = new Graph()) {
	      g.importGraphDef(graphDef);
	      try (Session s = new Session(g);		  
	          Tensor<Long> result = s.runner().feed("input_holder", image).feed("is_train", is_training).feed("dropout", dropout).fetch("digits_prediction").run().get(0).expect(Long.class)) {
	    	  final long[] rshape = result.shape();
	    	  int ndigits = (int)rshape[1];

	    	  return result.copyTo(new long[1][ndigits]);
	      }
	    }
	  }
	  static class GraphBuilder {
	    GraphBuilder(Graph g) {
	      this.g = g;
	    }
	    Output<Float> div(Output<Float> x, Output<Float> y) {
	      return binaryOp("Div", x, y);
	    }
	    <T> Output<T> sub(Output<T> x, Output<T> y) {
	      return binaryOp("Sub", x, y);
	    }
	    <T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
	      return binaryOp3("ResizeBilinear", images, size);
	    }
	    <T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
	      return binaryOp3("ExpandDims", input, dim);
	    }
	    <T> Output<T> expandDims1(Output<T> input, Output<Integer> dim) {
		      return binaryOp3("ExpandDims", input, dim);
		    }
	    <T> Output<T> transpose(Output<T> input, Output<Integer> dim){
	    	return binaryOp3("Transpose", input, dim);
	    }
	    <T, U> Output<U> cast(Output<T> value, Class<U> type) {
	      DataType dtype = DataType.fromClass(type);
	      return g.opBuilder("Cast", "Cast")
	          .addInput(value)
	          .setAttr("DstT", dtype)
	          .build()
	          .<U>output(0);
	    }
	    Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
	      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
	          .addInput(contents)
	          .setAttr("channels", channels)
	          .build()
	          .<UInt8>output(0);
	    }
	    <T> Output<T> constant(String name, Object value, Class<T> type) {
	      try (Tensor<T> t = Tensor.<T>create(value, type)) {
	        return g.opBuilder("Const", name)
	            .setAttr("dtype", DataType.fromClass(type))
	            .setAttr("value", t)
	            .build()
	            .<T>output(0);
	      }
	    }
	    Output<String> constant(String name, byte[] value) {
	    	return this.constant(name, value, String.class);
	    }
	    Output<Boolean> constant(String name, boolean value){
	    	return this.constant(name, value, Boolean.class);
	    }
	    Output<Integer> constant(String name, int value) {
	      return this.constant(name, value, Integer.class);
	    }
	    Output<Integer> constant(String name, int[] value) {
	      return this.constant(name, value, Integer.class);
	    }
	    Output<Float> constant(String name, float value) {
	      return this.constant(name, value, Float.class);
	    }
	    private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
	      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
	    }
	    private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
	      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
	    }
	    private Graph g;
	  }


}
