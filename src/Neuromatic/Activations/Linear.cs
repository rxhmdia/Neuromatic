using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// Linear activation function is a pass-through activation function. It is defined as y = x.
    /// </summary>
    public class Linear: ActivationFunction
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
            return context.Graph.Identity(input);
        }
    }
}