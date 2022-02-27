import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.RuleNode;
import org.antlr.v4.runtime.tree.TerminalNodeImpl;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.antlr.v4.runtime.tree.Tree;
import org.json.JSONArray;
import org.json.JSONObject;

public class ConvertJava {
  private static final int MAX_DEPTH = 1000;
  Vocabulary vocab;
  ArrayList<String> identifiersRuleNames =
      new ArrayList<String>(
          Arrays.asList(
              "IDENTIFIER",
              "localVar",
              "CHAR_LITERAL",
              "STRING_LITERAL",
              "BOOL_LITERAL",
              "NULL_LITERAL",
              "DECIMAL_LITERAL",
              "HEX_LITERAL",
              "OCT_LITERAL",
              "BINARY_LITERAL",
              "FLOAT_LITERAL",
              "HEX_FLOAT_LITERAL"));

  ArrayList<String> localVarContexts =
      new ArrayList<String>(
          Arrays.asList("variableDeclaratorId", "primary", "catchClause", "lambdaParameters"));

  List<String> ruleNames = null;

  private void setRuleNames(Parser recog) {
    String[] ruleNames = recog != null ? recog.getRuleNames() : null;
    this.ruleNames = ruleNames != null ? Arrays.asList(ruleNames) : null;
  }

  public String getRuleName(Tree t) {
    int ruleIndex = ((RuleNode) t).getRuleContext().getRuleIndex();
    return ruleNames.get(ruleIndex);
  }

  public void openWriter(String file) throws FileNotFoundException {
    writer = new PrintWriter(file);
  }

  public void openWriter(OutputStream stream) {
    writer = new PrintWriter(stream);
  }

  public void closeWriter() {
    writer.close();
  }

  private int totalFiles = 0;
  private int successFullFiles = 0;
  private int totalMethods = 0;

  public void serializeFile(String f, String startSymbol) {
    try {
      thisFileName = f;
      totalFiles++;
      stackDepth = 0;
      CharStream charStream = CharStreams.fromFileName(f);
      Lexer lexer = new JavaLexer(charStream);
      CommonTokenStream tokens = new CommonTokenStream(lexer);
      vocab = lexer.getVocabulary();
      Parser parser = new JavaParser(tokens);
      parser.setErrorHandler(new BailErrorStrategy());
      Method method = parser.getClass().getMethod(startSymbol);
      ParserRuleContext t = (ParserRuleContext) method.invoke(parser);
      parser.setBuildParseTree(false);
      setRuleNames(parser);
      JSONArray tree = getSerializedTree(t, tokens);
      if (tree.length() == 2) {
        tree = tree.getJSONArray(1);
      }
      successFullFiles++;
    } catch (Exception e) {
      System.err.println(
          "Total processed files, Successfully processed file, total methods: "
              + totalFiles
              + ", "
              + successFullFiles
              + ", "
              + totalMethods
              + ", "
              + thisFileName);
      System.err.println("Parser Exception: " + e);
      e.printStackTrace(); // so we can get the stack trace
    }
  }

  private String getLeadingOrTrailing(ParseTree tree, CommonTokenStream tokens, boolean isBefore) {
    int lastIndexOfToken;
    StringBuilder builder = new StringBuilder("");
    lastIndexOfToken = ((TerminalNodeImpl) tree).getSymbol().getTokenIndex();
    List<Token> ws = null;
    int HIDDEN = 1;
    if (lastIndexOfToken < 0) {
      return "";
    }
    if (isBefore) {
      ws = tokens.getHiddenTokensToLeft(lastIndexOfToken, HIDDEN);
    } else if (lastIndexOfToken >= 0 || lastIndexOfToken == -2) {
      ws = tokens.getHiddenTokensToRight(lastIndexOfToken, HIDDEN);
    }
    if (ws != null) {
      for (Token wst : ws) {
        builder.append(wst.getText());
      }
    }
    return builder.toString();
  }

  private boolean childHasLeaf;
  private String thisClassName;
  private String thisMethodName;
  private String thisFileName;
  private int beginLine, endLine;
  private PrintWriter writer;
  private int stackDepth = 0;

  private void setClassName(String thisRuleName, RuleContext t, int i) {
    if (thisRuleName.equals("classDeclaration") && i > 0) {
      ParseTree prev = t.getChild(i - 1);
      ParseTree curr = t.getChild(i);
      if (prev instanceof TerminalNodeImpl
          && curr instanceof TerminalNodeImpl
          && prev.getText().equals("class")) {
        Token thisToken = ((TerminalNodeImpl) curr).getSymbol();
        String ruleName = vocab.getDisplayName(thisToken.getType());
        if (ruleName.equals("IDENTIFIER")) {
          thisClassName = thisToken.getText();
          //System.err.println("Processing Class: " + thisClassName);
        }
      }
    }
  }

  private void dumpMethodAst(String thisRuleName, JSONArray simpleTree) {
    if (thisClassName != null && thisRuleName.equals("methodBody")) {
      if (simpleTree.length() == 2) {
        try {
          simpleTree = simpleTree.getJSONArray(1);
        } catch (Exception e) {
          System.err.println(simpleTree);
          e.printStackTrace();
          System.err.println("In " + thisFileName + ":" + thisClassName + ":" + thisMethodName+":"+beginLine);
          return;
        }
      }
      JSONObject tmp = new JSONObject();
      tmp.put("path", thisFileName);
      tmp.put("class", thisClassName);
      tmp.put("method", thisMethodName);
      tmp.put("beginline", beginLine);
      tmp.put("endline", endLine);
      tmp.put("ast", simpleTree);
      writer.println(tmp);
      writer.flush();
      totalMethods++;
      //System.err.println("Logged " + thisFileName + ":" + thisClassName + ":" + thisMethodName);
    }
  }

  private JSONArray getSerializedTree(RuleContext t, CommonTokenStream tokens) {
    stackDepth++;
    int n = t.getChildCount();
    boolean hasLeaf = false;
    if (n == 0 || stackDepth > MAX_DEPTH) {
      childHasLeaf = false;
      stackDepth--;
      return null;
    }
    String thisRuleName = getRuleName(t);
    String oldClassName = null;
    String oldMethodName = null;
    int oldBeginLine = 0;
    if (thisRuleName.equals("classDeclaration")) {
      oldClassName = thisClassName;
    }
    if (thisRuleName.equals("methodDeclaration")) {
      oldMethodName = thisMethodName;
      thisMethodName = ((TerminalNodeImpl) t.getChild(1)).getText();
      oldBeginLine = beginLine;
      beginLine = ((TerminalNodeImpl) t.getChild(1)).getSymbol().getLine();
    }
    JSONArray simpleTree = new JSONArray();
    JSONObject start = new JSONObject();
    start.put("label", "");
    simpleTree.put(0, start);
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      ParseTree tree = t.getChild(i);
      if (tree instanceof TerminalNodeImpl) {
        String s = tree.getText();
        if (!s.equals("<EOF>")) {
          Token thisToken = ((TerminalNodeImpl) tree).getSymbol();
          String ruleName = vocab.getDisplayName(thisToken.getType());
          String ws1 = getLeadingOrTrailing(tree, tokens, true);
          String ws2 = getLeadingOrTrailing(tree, tokens, false);
          JSONObject tok = new JSONObject();
          tok.put("token", s);
          tok.put("leading", ws1);
          tok.put("trailing", ws2);
          boolean isLeaf;
          if (identifiersRuleNames.contains(ruleName)) {
            if (localVarContexts.contains(thisRuleName)) {
              tok.put("var", true);
              // System.err.println(s);
            }
            isLeaf = true;
            sb.append("#");
            hasLeaf = true;
            setClassName(thisRuleName, t, i);
          } else {
            isLeaf = false;
            sb.append(s);
          }
          if (isLeaf) tok.put("leaf", isLeaf);
          tok.put("line", thisToken.getLine());
          endLine = thisToken.getLine();
          simpleTree.put(tok);
        }
      } else {
        JSONArray child = getSerializedTree((RuleContext) tree, tokens);
        if (child != null && child.length() > 0) {
          if (child.length() == 2) {
            simpleTree.put(child.get(1));
            sb.append(((JSONObject)child.get(0)).get("label"));
            hasLeaf = hasLeaf || childHasLeaf;
          } else if (!childHasLeaf
              && !((JSONObject)child.get(0)).get("label").equals("{}")) { // see the while(m.find()){} query
            sb.append(((JSONObject)child.get(0)).get("label"));
            for (int j = 1; j < child.length(); j++) {
              simpleTree.put(child.get(j));
            }
          } else {
            sb.append("#");
            hasLeaf = true;
            simpleTree.put(child);
          }
        }
      }
    }
    start.put("label", sb);
    simpleTree.put(0, start);
    childHasLeaf = hasLeaf;
    dumpMethodAst(thisRuleName, simpleTree);
    if (thisRuleName.equals("classDeclaration")) {
      thisClassName = oldClassName;
    }
    if (thisRuleName.equals("methodDeclaration")) {
      thisMethodName = oldMethodName;
      beginLine = oldBeginLine;
    }
    stackDepth--;
    return simpleTree;
  }

  public static void main(String args[]) throws IOException {
    ConvertJava p = new ConvertJava();

    if(args[1].equals("stdout")){
      p.openWriter(System.out);
    }
    else {
      p.openWriter(args[1]);
    }
    System.setErr(new PrintStream("parse_err.log"));

    File inputf = new File(args[2]);
    if(Files.isRegularFile(inputf.toPath())){
      if (args[2].endsWith(".java")) {
        p.serializeFile(args[2], args[0]);
      } 
      else if(inputf.toPath().toString().endsWith(".txt")){ //路径清单
        BufferedReader in = new BufferedReader(new FileReader(args[2]));
        String line;
        while((line = in.readLine()) != null){
          p.serializeFile(line, args[0]);
        }
        in.close();
      }
    }
    else {
      Files.walk(Paths.get(args[2]))
          .filter(path -> !Files.isDirectory(path) && path.toString().endsWith(".java"))
          .forEach(path -> p.serializeFile(path.normalize().toString(), args[0]));
    }
    p.closeWriter();
  }
}
