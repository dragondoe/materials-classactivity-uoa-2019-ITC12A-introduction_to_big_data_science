import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class GenerateSVMData {

    public static void main(String[] args) throws IOException {

        MyDataReader readfile = new MyDataReader();
        List<String> documents1 = new ArrayList<String>();
        double div_rate = 0.75;  // can be modified.

        if (args.length != 3 ) {
	    System.out.println("Usage --> java .;igo-0.4.5 GenerateSVMData InputData OutputData [Divide-Rate]<Enter>");

	    System.out.print("Now args are ");
	    for(int i = 0; i < args.length; i++) System.out.print(args[i] + " ");
	    System.out.println();
	    
	    System.exit(0);
	    }
	documents1 = readfile.read(args[0]);

        if ( args.length == 3) div_rate = Double.parseDouble(args[2]);
        MyDataWriter writefile = new MyDataWriter();
        writefile.write(args[1], documents1, div_rate);
   }

}
