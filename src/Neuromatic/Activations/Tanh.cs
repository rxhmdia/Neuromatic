using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// The tanh function looks similar to the <see cref="Sigmoid"/> but has a much steeper gradient
    /// in the domain -2 + 2 for the input. It performs better in deeper neural networks, because it suffers
    /// a little less from the vanishing gradient problem.
    /// </summary>
    public class Tanh: ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        /// <param name="input">Input for the activation function</param>
        /// <returns>Returns the compiled activation function</returns>
        public override TFOutput Compile(ModelCompilationContext context, TFOutput input)
        {
            return context.Graph.Tanh(input);
        }
    }
}