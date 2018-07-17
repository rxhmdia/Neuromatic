using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Losses
{
    /// <summary>
    /// <para>
    /// The negative log-likelihood loss function is very similar to the categorical cross entropy function.
    /// It expects the output layer to have a softmax activation function. It calculates the likelihood that
    /// the output produces a probability distribution that is close to the expected output.
    /// </para>
    /// <para>
    /// The negative log-likelihood is defined as loss = -sum(log(pred)*target)
    /// </para>
    /// <para>The predicted probabilities are clipped to prevent this formula
    /// from producing loss values into infinity.</para>
    /// </summary>
    public class NegativeLogLikelihood : LossFunction
    {
        private readonly double _epsilon;

        /// <summary>
        /// Initializes a new instance of <see cref="NegativeLogLikelihood"/>
        /// </summary>
        /// <param name="epsilon">
        /// A predefined value by which inputs should be clipped to prevent overflows.
        /// This value should be really small.
        /// </param>
        public NegativeLogLikelihood(double epsilon = 1e-07)
        {
            _epsilon = epsilon;
        }

        /// <summary>
        /// Compiles the loss function
        /// </summary>
        /// <param name="context">Compilation context to use</param>
        /// <param name="predictions">The output of the graph producing predictions</param>
        /// <param name="targets">The output of the graph containing the targets</param>
        /// <returns>Returns the compiled loss function</returns>
        public override TFOutput Compile(ModelCompilationContext context, TFOutput predictions, TFOutput targets)
        {
            // Formula: loss = -sum(log(pred) * target)
            var graph = context.Graph;

            // Values should be clipped to be between epsilon and 1-epsilon
            // to prevent arithmic overflows into infinity.
            var clipped = graph.ClipByValue(predictions, 
                graph.Const(_epsilon), 
                graph.Const(1 - _epsilon));

            return graph.Neg(graph.ReduceSum(graph.Mul(targets, graph.Log(clipped))));
        }
    }
}
