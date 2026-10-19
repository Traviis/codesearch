using ScipCsharp;
using Xunit;

namespace ScipCsharp.Tests;

/// <summary>
/// PathGuard invariants (Aikido SAST 30640677): caller-supplied paths must
/// resolve absolutely, and derived paths must stay inside their base
/// directory without unresolved '..' segments.
/// </summary>
public class PathGuardTests
{
    [Fact]
    public void Validate_ReturnsAbsolutePath_ForExistingFile()
    {
        var path = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "x.txt"));
        Assert.Equal(path, PathGuard.Validate(path));
    }

    [Fact]
    public void Validate_RejectsRelativePath()
    {
        Assert.ThrowsAny<ArgumentException>(() => PathGuard.Validate("relative/file.txt"));
    }

    [Fact]
    public void Inside_AcceptsPathUnderBase()
    {
        var baseDir = Path.GetTempPath();
        var candidate = Path.Combine(baseDir, "sub", "file.json");
        Assert.Equal(Path.GetFullPath(candidate), PathGuard.Inside(baseDir, candidate));
    }

    [Fact]
    public void Inside_AcceptsBaseItself()
    {
        var baseDir = Path.GetFullPath(Path.GetTempPath());
        Assert.Equal(baseDir, PathGuard.Inside(baseDir, baseDir));
    }

    [Fact]
    public void Inside_RejectsEscapeToSibling()
    {
        var baseDir = Path.GetFullPath(Path.GetTempPath());
        var sibling = Path.Combine(Path.GetTempPath(), "..", "elsewhere");
        Assert.ThrowsAny<ArgumentException>(() => PathGuard.Inside(baseDir, sibling));
    }

    [Fact]
    public void Inside_RejectsUnresolvedDotDotSegment()
    {
        var baseDir = Path.GetFullPath(Path.GetTempPath());
        // Normalizes INSIDE baseDir (containment passes) but still carries
        // a '..' segment, so this reaches the segment check specifically.
        var candidate = Path.Combine(baseDir, "sub", "..", "file.json");
        Assert.ThrowsAny<ArgumentException>(() => PathGuard.Inside(baseDir, candidate));
    }
}
