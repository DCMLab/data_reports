"""Copied and adapted from https://github.com/cophi-wue/pydelta/tree/35b0614 which was used in
Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017). Understanding and
explaining Delta measures for authorship attribution. Digital Scholarship in the Humanities, 32(suppl_2), ii4–ii16.
https://doi.org/10.1093/llc/fqx023
"""
from __future__ import annotations

import json
from typing import List, Mapping

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from numpy import typing as npt
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from scipy.spatial.distance import squareform


class Metadata(Mapping):
    """
    A metadata record contains information about how a particular object of the
    pyDelta universe has been constructed, or how it will be manipulated.

    Metadata fields are simply attributes, and they can be used as such.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new metadata instance. Arguments will be passed on to :meth:`update`.

        Examples:
            >>> m = Metadata(lower_case=True, sorted=False)
            >>> Metadata(m, sorted=True, words=5000)
            Metadata(lower_case=True, sorted=True, words=5000)
        """
        self.update(*args, **kwargs)

    def _update_from(self, d):
        """
        Internal helper to update inner dictionary 'with semantics'. This will
        append rather then overwrite existing md fields if they are in a
        specified list. Clients should use :meth:`update` or the constructor
        instead.

        Args:
            d (dict): Dictionary to update from.
        """
        if isinstance(d, dict):
            appendables = ("normalization",)
            d2 = dict(d)

            for field in appendables:
                if field in d and field in self.__dict__:
                    d2[field] = self.__dict__[field] + d[field]

            self.__dict__.update(d2)
        else:
            self.__dict__.update(d)

    # maybe inherit from mappingproxy?
    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def update(self, *args, **kwargs):
        """
        Updates this metadata record from the arguments. Arguments may be:

        * other :class:`Metadata` instances
        * objects that have ``metadata`` attribute
        * JSON strings
        * stuff that :class:`dict` can update from
        * key-value pairs of new or updated metadata fields
        """
        for arg in args:
            if isinstance(arg, Metadata):
                self._update_from(arg.__dict__)
            elif "metadata" in dir(arg) and isinstance(arg.metadata, Metadata):
                self._update_from(arg.metadata.__dict__)
            elif isinstance(arg, str):
                self._update_from(json.loads(arg))
            elif arg is not None:
                self._update_from(arg)
        self._update_from(kwargs)

    @staticmethod
    def metafilename(filename):
        """
        Returns an appropriate metadata filename for the given filename.

        >>> Metadata.metafilename("foo.csv")
        'foo.csv.meta'
        >>> Metadata.metafilename("foo.csv.meta")
        'foo.csv.meta'
        """
        if filename.endswith(".meta"):
            return filename
        return filename + ".meta"

    @classmethod
    def load(cls, filename):
        """
        Loads a metadata instance from the filename identified by the argument.

        Args:
            filename (str): The name of the metadata file, or of the file to which a sidecar metadata filename exists
        """
        metafilename = cls.metafilename(filename)
        with open(metafilename, "rt", encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d, dict):
                return cls(**d)
            else:
                raise TypeError(
                    "Could not load metadata from {file}: \n"
                    "The returned type is a {type}".format(
                        file=metafilename, type=type(d)
                    )
                )

    def save(self, filename, **kwargs):
        """
        Saves the metadata instance to a JSON file.

        Args:
            filename (str): Name of the metadata file or the source file
            **kwargs: are passed on to :func:`json.dump`
        """
        metafilename = self.metafilename(filename)
        with open(metafilename, "wt", encoding="utf-8") as f:
            json.dump(self.__dict__, f, **kwargs)

    def __repr__(self):
        return (
            type(self).__name__
            + "("
            + ", ".join(
                str(key) + "=" + repr(self.__dict__[key])
                for key in sorted(self.__dict__.keys())
            )
            + ")"
        )

    def to_json(self, **kwargs):
        """
        Returns a JSON string containing this metadata object's contents.

        Args:
            **kwargs: Arguments passed to :func:`json.dumps`
        """
        return json.dumps(self.__dict__, **kwargs)


class Clustering:
    """
    Represents a hierarchical clustering.

    Note:
        This is subject to refactoring once we implement more clustering
        methods
    """

    def __init__(self, distance_matrix, method="ward", **kwargs):
        self.metadata = Metadata(
            distance_matrix.metadata, cluster_method=method, **kwargs
        )
        self.distance_matrix = distance_matrix
        self.describer = distance_matrix.document_describer
        self.method = method
        self.linkage = self._calc_linkage()

    def _calc_linkage(self):
        if self.method == "ward":
            return ward(squareform(self.distance_matrix, force="tovector"))
        else:
            return linkage(
                squareform(self.distance_matrix), method=self.method, metric="euclidean"
            )

    # def fclustering(self):
    #     """
    #     Returns a default flat clustering from the hierarchical version.
    #
    #     This method uses the :class:`DocumentDescriber` to determine the
    #     groups, and uses the number of groups as a maxclust criterion.
    #
    #     Returns:
    #         FlatClustering: A properly initialized representation of the flat
    #         clustering.
    #     """
    #     flat = FlatClustering(self.distance_matrix, metadata=self.metadata,
    #                           flattening='maxclust')
    #     flat.set_clusters(sch.fcluster(self.linkage, flat.group_count,
    #                                    criterion="maxclust"))
    #     return flat


class TableDocumentDescriber:
    """
    A document decriber that takes groups and item labels from an external
    table.
    """

    def __init__(
        self, table, group_col="corpus", name_col="piece", dialect="excel", **kwargs
    ):
        """
        Args:
            table (str or pandas.DataFrame):
                A table with metadata that describes the documents of the
                corpus, either a :class:`pandas.DataFrame` or path or IO to a
                CSV file. The tables index (or first column for CSV files)
                contains the document ids that are returned by the
                :class:`FeatureGenerator`. The columns (or first row) contains
                column labels.
            group_col (str):
                Name of the column in the table that contains the names of the
                groups. Will be used, e.g., for determining the ground truth
                for cluster evaluation, and for coloring the dendrograms.
            name_col (str):
                Name of the column in the table that contains the names of the
                individual items.
            dialect (str or :class:`csv.Dialect`):
                CSV dialect to use for reading the file.
            **kwargs:
                Passed on to :func:`pandas.read_table`.
        Raises:
            ValueError: when arguments inconsistent
        See:
            pandas.read_table
        """
        if isinstance(table, pd.DataFrame):
            self.table = table
        else:
            self.table = pd.read_table(
                table, header=0, index_col=0, dialect=dialect, **kwargs
            )
        self.group_col = group_col
        self.name_col = name_col

        if not (group_col in self.table.columns):
            raise ValueError(
                "Given group column {} is not in the table: {}".format(
                    group_col, self.table.columns
                )
            )
        if not (name_col in self.table.columns):
            raise ValueError(
                "Given name column {} is not in the table: {}".format(
                    name_col, self.table.columns
                )
            )

    def group_name(self, document_name):
        try:
            return self.table.at[document_name, self.group_col]
        except KeyError:
            return document_name.split(", ")[0]

    def item_name(self, document_name):
        try:
            return self.table.at[document_name, self.name_col]
        except KeyError:
            return ", ".join(document_name.split(", ")[1:])

    def group_label(self, document_name):
        """
        Returns a (maybe shortened) label for the group, for display purposes.

        The default implementation just returns the :meth:`group_name`.
        """
        return self.group_name(document_name)

    def item_label(self, document_name):
        """
        Returns a (maybe shortened) label for the item within the group, for
        display purposes.

        The default implementation just returns the :meth:`item_name`.
        """
        return self.item_name(document_name)

    def label(self, document_name):
        """
        Returns a label for the document (including its group).
        """
        return self.group_label(document_name) + ", " + self.item_label(document_name)

    def groups(self, documents=None):
        """
        Returns the names of all groups of the given list of documents.
        """
        if documents is None:
            return set(self.table[self.group_col])
        return {self.group_name(document) for document in documents}


class Dendrogram:
    """
    Creates a dendrogram representation from a hierarchical clustering.

    This is a wrapper around, and an improvement to, :func:`sch.dendrogram`,
    tailored for the use in pydelta.

    Args:
        clustering (Clustering): A hierarchical clustering.
        describer (DocumentDescriber): Document describer used for determining
            the groups and the labels for the documents used (optional). By
            default, the document describer inherited from the clustering is
            used.
        ax (mpl.axes.Axes): Axes object to draw on. Uses pyplot default axes if
            not provided.
        orientation (str): Orientation of the dendrogram. Currently, only
            "right" is supported (default).
        font_size: Font size for the label, in points. If not provided,
            :func:`sch.dendrogram` calculates a default.
        link_color (str): The color used for the links in the dendrogram, by
            default ``k`` (for black).
        title (str): a title that will be printed on the plot. The string may
           be a template string as supported by :meth:`str.format_map` with
           metadata field names in curly braces, it will be evaluated against
           the clustering's metadata. If you pass ``None`` here, no title will
           be added.

    Notes:
        The dendrogram will be painted by matplotlib / pyplot using the default
        styles, which means you can use, e.g., :module:`seaborn` to influence
        the overall design of the image.

        :class:`Dendrogram` handles coloring differently than
        :func:`sch.dendrogram`: It will color the document labels according to
        the pre-assigned grouping (e.g., by author). To do so, it will build on
        matplotlib's default color_cycle, and it will rotate, so if you need
        more colors, adjust the color_cycle accordingly.
    """

    def __init__(
        self,
        linkage: npt.NDArray,
        describer: TableDocumentDescriber,
        documents: List[str],
        ax=None,
        orientation="left",
        font_size=None,
        link_color=None,
        title=None,
        xlabel="Delta: {delta_title}, {words} most frequent {features}",
    ):
        # self.clustering = clustering
        self.linkage = linkage
        # self.metadata = clustering.metadata
        self.describer = describer
        self.documents = documents
        self.orientation = orientation
        self._init_colormap()

        plt.clf()
        link_color_func = None if link_color is None else lambda k: link_color
        self.dendro_data = dendrogram(
            self.linkage,
            orientation=orientation,
            labels=self.documents,
            leaf_rotation=0 if orientation == "left" else 90,
            leaf_font_size=font_size,
            ax=ax,
            link_color_func=link_color_func,
        )

        # Now redo the author labels. To do so, we map a color to each author
        # (using the describer) and then
        self.ax = plt.gca() if ax is None else ax
        self.fig = plt.gcf()
        self._relabel_axis()
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        plt.tight_layout(pad=2)

    def link_color_func(self, k):
        print(k)
        return "k"

    def _init_colormap(self):
        groups = self.describer.groups(self.documents)
        props = mpl.rcParams["axes.prop_cycle"]
        self.colormap = {x: y["color"] for x, y in zip(groups, props())}
        self.colorlist = [
            self.colormap[self.describer.group_name(doc)] for doc in self.documents
        ]
        return self.colormap

    def _relabel_axis(self):
        if self.orientation == "left":
            labels = self.ax.get_ymajorticklabels()
        else:
            labels = self.ax.get_xmajorticklabels()
        display_labels = []
        for label in labels:
            group = self.describer.group_name(label.get_text())
            label.set_color(self.colormap[group])
            display_label = self.describer.label(label.get_text())
            label.set_text(display_label)  # doesn't really set the labels
            display_labels.append(display_label)
        if self.orientation == "left":
            self.ax.set_yticklabels(display_labels)
        else:
            self.ax.set_xticklabels(display_labels)

    def show(self):
        plt.show()

    def save(self, fname, **kwargs):
        self.fig.savefig(fname, **kwargs)
