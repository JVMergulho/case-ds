class RankCountVectorizer(object):
    """Vectorizes categorical variables by rank of magnitude.
    This is a hybrid between label vectorizing (assigning every variable a unique ID)
    and count vectorizing (replacing variables with their count in training set).
    Categorical variables are ranked by their count in train set. If a never-before-seen
    variable is in test set, it gets assigned `1`, and is treated like a rare variable in
    the train set. NaN's can be treated as specifically encoded, for instance with `-1`,
    or they can be set to `0`, meaning some algorithms will treat them as missing / ignore
    them. Linear algorithms should be able to work with labelcount encoded variables.
    They basically get treated as: how popular are these variables?
    Example:
        |cat|
        -----
        a
        a
        a
        a
        b
        c
        c
        NaN
        vectorizes to:
        |cat|
        -----
        3
        3
        3
        3
        1
        2
        2
        -1
    Attributes:
        verbose: An integer specifying level of verbosity. Default is `0`.
        set_nans: An integer for filling NaN values. Default is `-1`.
    """

    def __init__(self, verbose=0, set_nans=-999, min_count=40):
        self.verbose = verbose
        self.set_nans = set_nans
        self.min_count = min_count

    def __repr__(self):
        return "RankCountVectorizer(verbose=%s, set_nans=%s)" % (
            self.verbose,
            self.set_nans,
        )

    def fit(self, df, cols=[]):
        """Fits a vectorizer to a dataframe.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        """
        if self.verbose > 0:
            print(
                "Labelcount fitting columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )

        vec = {}
        for col in cols:
            vec[col] = {}

        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            d = df[col].value_counts()
            d = d[d > self.min_count]
            for i, k in enumerate(sorted(d.to_dict(), key=d.get)):
                vec[col][k] = i + 1
            vec[col][-999] = self.set_nans
        self.vec = vec

    def transform(self, df, cols=[]):
        """Transforms a dataframe with a vectorizer.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        Raises:
            AttributeError: Transformation was attempted before fitting the vectorizer.
        """
        try:
            self.vec
        except AttributeError:
            import sys

            sys.exit(
                "AttributeError. `self.vec` is not set. Use .fit() before transforming."
            )

        if self.verbose > 0:
            print(
                "Labelcount transforming columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )

        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            df[col].fillna(-999, inplace=True)
            df[col] = df[col].apply(
                lambda x: self.vec[col][x] if x in self.vec[col] else 1
            )
        return df

    def fit_transform(self, df, cols=[]):
        """Calls fit then calls transform in one line.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        """
        self.fit(df, cols)
        return self.transform(df, cols)