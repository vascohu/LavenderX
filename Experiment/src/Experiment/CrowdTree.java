package Experiment;

import org.apache.commons.math3.linear.RealVector;

import java.util.Date;
import java.util.List;
import java.util.concurrent.ThreadPoolExecutor;

/* The tree approximates the value function with a d-depth tree.
 * We build the tree with depth-first search.
 * Created by Zehong on 3/22/2017 0022.
 */
class CrowdTree {
    // The depth of the tree
    private final static int depth = 5;
    // The width of the sparse tree
    private final static int width = 1;
    // The discount of RL
    private final static double gamma = 0.9;
    // The root
    final CrowdNode root;

    public Date t1;

    CrowdTree(DVState s_vec, Prob_Model model, Obj_Function obj, List<Action> available_action, ThreadPoolExecutor executor)
    {
        root = new CrowdNode(null, null, 0, obj, false, 1.0);
        root.crowdModel = model;
        root.objValue = obj.getObjValue(model);
        TreeGrow(root, Math.min(depth, available_action.size()), s_vec, available_action, executor);
    }

    CrowdTree(DVState s_vec, Prob_Model model, Obj_Function obj, List<Action> available_action, ThreadPoolExecutor executor, boolean Sparse_Or_Not)
    {
        t1 = new Date();
        root = new CrowdNode(null, null, 0, obj, false, 1.0, Sparse_Or_Not);
        root.crowdModel = model;
        root.objValue = obj.getObjValue(model);
        TreeGrow(root, Math.min(depth, available_action.size()), s_vec, available_action, executor);
    }

    private void TreeGrow(CrowdNode p_node, int d, DVState s_vec, List<Action> available_action, ThreadPoolExecutor executor) {
        if(d!=0)
        {
            Date t2 = new Date();
            System.out.println("Level "+d+" Before Children: "+(t2.getTime()-t1.getTime()));
            p_node.Gene_Children(width, s_vec, available_action, executor);
            t2 = new Date();
            System.out.println("Level "+d+" After Children: "+(t2.getTime()-t1.getTime()));
            double action_value = -Double.MAX_VALUE;
            for(List<CrowdNode> action_node: p_node.childNodes)
            {
                Action parent_action = action_node.get(0).parentAction;
                int index_of_action = available_action.indexOf(parent_action);
                available_action.remove(index_of_action);
                RealVector label_prob = p_node.crowdModel.getLabelProb(parent_action);
                double this_action_value = 0;
                for(CrowdNode node: action_node)
                {
                    s_vec.setEntry(parent_action.i, parent_action.j, node.observedLabel);
                    TreeGrow(node, d-1, s_vec, available_action, executor);
                    this_action_value += label_prob.getEntry(node.observedLabel-1)*node.objValue;
                }
                s_vec.setEntry(parent_action.i, parent_action.j, 0);
                available_action.add(index_of_action, parent_action);
                if (this_action_value>action_value)
                {
                    action_value = this_action_value;
                }
            }
            p_node.objValue += gamma*(action_value - p_node.objValue);
            t2 = new Date();
            System.out.println("Level "+d+" Final: "+(t2.getTime()-t1.getTime()));
        }
    }
}