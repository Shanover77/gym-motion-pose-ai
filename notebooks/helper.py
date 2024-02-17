def cosine_interpolator(df, title, factor=0.2):
    x = df.index.values

    # Interpolate using linear interpolation
    interp_func_x = interp1d(x, df["X"], kind="linear", fill_value="extrapolate")
    interp_func_y = interp1d(x, df["Y"], kind="linear", fill_value="extrapolate")

    # Generate new x values for smoother motion
    new_x = np.arange(x[0], x[-1], factor)

    # Use cosine function to remove unrealistic motion caused by an outlier
    smooth_x = interp_func_x(new_x) + factor * np.cos(new_x * np.pi / (2 * x[-1]))
    smooth_y = interp_func_y(new_x) + factor * np.cos(new_x * np.pi / (2 * x[-1]))

    # Create a DataFrame of smoothed coordinates
    smoothed_coordinates = pd.DataFrame(
        {"Frame Index": new_x, "Smoothed_X": smooth_x, "Smoothed_Y": smooth_y}
    )

    return smoothed_coordinates
