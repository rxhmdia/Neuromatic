using System;

namespace Neuromatic
{
    /// <summary>
    /// Indicates a problem compiling parts of the model.
    /// Make sure to check the message for specific details on what failed.
    /// </summary>
    public class ModelCompilationException : Exception
    {
        /// <summary>
        /// Initializes a new instance of <see cref="ModelCompilationException"/>
        /// </summary>
        /// <param name="message">Message to display</param>
        public ModelCompilationException(string message) : base(message)
        {
        }
    }
}
